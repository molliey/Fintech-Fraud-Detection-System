"""
visa_fraud/data_pipeline.py
────────────────────────────
Realistic transaction data generation and preprocessing pipeline.

In production this connects to:
  - Kafka topic: visa.transactions.realtime (Confluent Kafka)
  - Hive tables: visa_dw.transactions_raw, visa_dw.cardholder_profiles
  - Feature Store: Feast / internal Visa FS
  - Redis: real-time velocity counters

Here we simulate that data with statistically accurate distributions
matching public Visa/Mastercard fraud research papers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# SECTION 1: Realistic Transaction Simulator
# ──────────────────────────────────────────────────────────────
class VisaTransactionSimulator:
    """
    Generates transactions that mirror the statistical properties
    described in:
      - Visa's public fraud reports
      - IEEE TNNLS "Credit Card Fraud Detection" benchmark papers
      - The European card fraud dataset characteristics

    Key realistic properties:
      - Highly imbalanced (0.17% fraud per Visa 2023 report)
      - Fraud clusters in time (account takeover patterns)
      - Card-not-present fraud dominates (76% of fraud)
      - Cross-border fraud overrepresented
    """

    # Real-world Merchant Category Code risk tiers (based on Visa public data)
    HIGH_RISK_MCC = {
        5912,  # Drug stores / pharmacies
        5999,  # Misc retail
        7995,  # Gambling
        6051,  # Quasi-cash
        4814,  # Telecom
        5047,  # Medical supplies
        7011,  # Hotels (CNP)
        4722,  # Travel agencies
        5732,  # Electronics
        5065,  # Electronic components
    }

    MCC_CATEGORIES = {
        "grocery":      (5411, 0.22),
        "restaurant":   (5812, 0.15),
        "gas":          (5541, 0.10),
        "retail":       (5999, 0.18),
        "online":       (5965, 0.15),
        "travel":       (4722, 0.06),
        "healthcare":   (5912, 0.05),
        "entertainment":(7995, 0.04),
        "financial":    (6051, 0.03),
        "telecom":      (4814, 0.02),
    }

    HIGH_RISK_COUNTRIES = {
        "NG", "RO", "UA", "BY", "PK", "BD", "VN", "ID", "RU", "KZ"
    }

    def __init__(self, n_cardholders: int = 5_000, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_cardholders = n_cardholders
        self._build_cardholder_profiles()
        self._build_merchant_profiles()

    def _build_cardholder_profiles(self):
        """Create realistic cardholder behavioral profiles."""
        rng = self.rng
        n = self.n_cardholders

        self.cardholder_profiles = pd.DataFrame({
            "cardholder_id":    [f"CH{i:07d}" for i in range(n)],
            "home_country":     rng.choice(
                ["US", "UK", "CA", "DE", "FR", "AU", "JP"],
                size=n, p=[0.45, 0.12, 0.10, 0.09, 0.08, 0.08, 0.08]
            ),
            "segment":          rng.choice(
                ["PREMIUM", "STANDARD", "BASIC"],
                size=n, p=[0.20, 0.55, 0.25]
            ),
            # Typical spend: log-normal, differs by segment
            "typical_spend_mu": rng.uniform(3.0, 5.5, n),
            "typical_spend_sigma": rng.uniform(0.5, 1.5, n),
            # Typical hour of day (Gaussian)
            "typical_hour_mu":  rng.uniform(9.0, 20.0, n),
            "typical_hour_sigma": rng.uniform(2.0, 4.0, n),
            # Most frequent merchant category
            "primary_category": rng.choice(
                list(self.MCC_CATEGORIES.keys()),
                size=n,
                p=[v[1] for v in self.MCC_CATEGORIES.values()]
            ),
            "card_age_days":    rng.integers(30, 2000, n),
            "account_age_days": rng.integers(180, 5000, n),
        })
        self.cardholder_profiles.set_index("cardholder_id", inplace=True)

    def _build_merchant_profiles(self, n_merchants: int = 500):
        """Create merchant profiles with realistic fraud rates."""
        rng = self.rng
        categories = list(self.MCC_CATEGORIES.keys())
        mccs = [self.MCC_CATEGORIES[c][0] for c in categories]

        cats = rng.choice(categories, size=n_merchants,
                          p=[v[1] for v in self.MCC_CATEGORIES.values()])
        self.merchant_profiles = pd.DataFrame({
            "merchant_id":   [f"MER{i:06d}" for i in range(n_merchants)],
            "category":      cats,
            "mcc":           [self.MCC_CATEGORIES[c][0] for c in cats],
            "country":       rng.choice(
                ["US", "UK", "CA", "DE", "FR", "NG", "RO", "CN"],
                size=n_merchants, p=[0.40,0.10,0.08,0.08,0.07,0.07,0.10,0.10]
            ),
            "avg_ticket":    np.abs(rng.lognormal(3.5, 1.0, n_merchants)),
            # Baseline merchant-level fraud rate (public data: ~0.1-2%)
            "base_fraud_rate": np.clip(rng.lognormal(-4.5, 1.0, n_merchants), 0.001, 0.05),
            "age_days":      rng.integers(30, 3650, n_merchants),
            "is_card_not_present": rng.choice([0, 1], n_merchants, p=[0.45, 0.55]),
        })
        self.merchant_profiles.set_index("merchant_id", inplace=True)

    def generate_transactions(
        self,
        n_transactions: int = 100_000,
        fraud_rate: float = 0.0172,  # Visa 2023 reported rate
        start_date: datetime = None,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Generate a realistic transaction dataset.

        Fraud patterns simulated:
        1. Account Takeover (ATO) - velocity spikes, new device
        2. Card Not Present (CNP) - online, high amount, foreign
        3. Lost/Stolen Physical Card - small test transactions first
        4. Synthetic Identity - new account, high spend immediately
        5. Friendly Fraud - cardholder disputes legitimate charge
        """
        logger.info(f"Generating {n_transactions:,} transactions "
                    f"(fraud rate: {fraud_rate:.2%})...")

        rng = self.rng
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)

        n_fraud = int(n_transactions * fraud_rate)
        n_legit = n_transactions - n_fraud

        legit_df = self._generate_legitimate(n_legit, start_date, days)
        fraud_df = self._generate_fraud(n_fraud, start_date, days)

        df = pd.concat([legit_df, fraud_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df["txn_id"] = [f"TXN{i:010d}" for i in range(len(df))]

        logger.info(
            f"Dataset: {len(df):,} rows | "
            f"Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.3%}) | "
            f"Date range: {df['txn_datetime'].min().date()} → "
            f"{df['txn_datetime'].max().date()}"
        )
        return df

    def _generate_legitimate(self, n: int, start: datetime, days: int) -> pd.DataFrame:
        rng = self.rng
        ch_ids = rng.choice(self.cardholder_profiles.index, n)
        profiles = self.cardholder_profiles.loc[ch_ids].reset_index()
        mer_ids = rng.choice(self.merchant_profiles.index, n)
        merchants = self.merchant_profiles.loc[mer_ids].reset_index()

        # Amounts drawn from cardholder-specific log-normal
        amounts = np.array([
            max(0.50, rng.lognormal(mu, sig))
            for mu, sig in zip(profiles["typical_spend_mu"], profiles["typical_spend_sigma"])
        ])

        # Hours drawn from cardholder's typical hour distribution
        hours = np.clip(
            rng.normal(profiles["typical_hour_mu"], profiles["typical_hour_sigma"]).astype(int),
            0, 23
        )

        # Timestamps: weighted toward business hours
        offsets = rng.uniform(0, days * 86400, n).astype(int)
        timestamps = [start + timedelta(seconds=int(o)) for o in offsets]

        return pd.DataFrame({
            "cardholder_id":         ch_ids,
            "merchant_id":           mer_ids,
            "merchant_category":     merchants["category"].values,
            "mcc":                   merchants["mcc"].values,
            "amount":                np.round(amounts, 2),
            "currency":              "USD",
            "txn_datetime":          timestamps,
            "hour_of_day":           hours,
            "day_of_week":           [t.weekday() for t in timestamps],
            "is_weekend":            [(t.weekday() >= 5) * 1 for t in timestamps],
            "card_present":          rng.choice([0, 1], n, p=[0.38, 0.62]),
            "is_emv_chip":           rng.choice([0, 1], n, p=[0.10, 0.90]),
            "is_contactless":        rng.choice([0, 1], n, p=[0.45, 0.55]),
            "is_3ds_authenticated":  rng.choice([0, 1], n, p=[0.25, 0.75]),
            "cardholder_country":    profiles["home_country"].values,
            "merchant_country":      merchants["country"].values,
            "is_domestic":           (profiles["home_country"].values == merchants["country"].values).astype(int),
            "card_age_days":         profiles["card_age_days"].values,
            "device_fingerprint_match": rng.choice([0, 1], n, p=[0.03, 0.97]),
            "is_new_device":         rng.choice([0, 1], n, p=[0.96, 0.04]),
            "txn_count_1h":          rng.poisson(1.2, n),
            "txn_count_6h":          rng.poisson(2.5, n),
            "txn_count_24h":         rng.poisson(5.0, n),
            "txn_count_7d":          rng.poisson(18, n),
            "amount_sum_1h":         np.round(np.abs(rng.lognormal(3.0, 1.2, n)), 2),
            "amount_sum_24h":        np.round(np.abs(rng.lognormal(4.5, 1.2, n)), 2),
            "unique_merchants_24h":  rng.integers(1, 6, n),
            "unique_countries_7d":   rng.choice([1, 2, 3], n, p=[0.88, 0.09, 0.03]),
            "declined_count_24h":    rng.choice([0, 1, 2], n, p=[0.92, 0.06, 0.02]),
            "days_since_last_txn":   rng.integers(0, 10, n),
            "distance_from_home_km": np.abs(rng.exponential(15, n)),
            "impossible_travel_flag":rng.choice([0, 1], n, p=[0.998, 0.002]),
            "ip_country_mismatch":   rng.choice([0, 1], n, p=[0.96, 0.04]),
            "is_high_risk_country":  merchants["country"].isin(self.HIGH_RISK_COUNTRIES).astype(int).values,
            "is_high_risk_mcc":      merchants["mcc"].isin(self.HIGH_RISK_MCC).astype(int).values,
            "merchant_base_fraud_rate": merchants["base_fraud_rate"].values,
            "is_card_not_present":   merchants["is_card_not_present"].values,
            "billing_address_change_flag": rng.choice([0, 1], n, p=[0.99, 0.01]),
            "days_since_last_change": rng.integers(10, 500, n),
            "is_recurring":          rng.choice([0, 1], n, p=[0.80, 0.20]),
            "velocity_score":        np.clip(rng.beta(1.5, 8, n), 0, 1),
            "is_fraud":              0,
        })

    def _generate_fraud(self, n: int, start: datetime, days: int) -> pd.DataFrame:
        """
        Generate fraud transactions across 5 realistic fraud patterns.
        Each pattern has distinct statistical fingerprint.
        """
        rng = self.rng

        # Distribute fraud across patterns (based on industry reports)
        pattern_counts = {
            "cnp_fraud":            int(n * 0.38),  # Card-not-present: 38%
            "account_takeover":     int(n * 0.28),  # ATO: 28%
            "lost_stolen":          int(n * 0.18),  # Lost/stolen: 18%
            "synthetic_identity":   int(n * 0.11),  # Synthetic ID: 11%
            "friendly_fraud":       n - int(n*0.95), # Friendly: 5%
        }

        frames = []

        # ── Pattern 1: Card-Not-Present Fraud ──
        n1 = pattern_counts["cnp_fraud"]
        ch_ids = rng.choice(self.cardholder_profiles.index, n1)
        mer_ids = rng.choice(self.merchant_profiles.index, n1)
        profiles = self.cardholder_profiles.loc[ch_ids].reset_index()
        merchants = self.merchant_profiles.loc[mer_ids].reset_index()
        offsets = rng.uniform(0, days * 86400, n1).astype(int)
        timestamps = [start + timedelta(seconds=int(o)) for o in offsets]

        frames.append(pd.DataFrame({
            "cardholder_id":         ch_ids,
            "merchant_id":           mer_ids,
            "merchant_category":     rng.choice(["online", "travel", "financial"], n1, p=[0.60, 0.25, 0.15]),
            "mcc":                   rng.choice([5965, 4722, 6051], n1, p=[0.60, 0.25, 0.15]),
            "amount":                np.round(np.abs(rng.lognormal(5.8, 1.2, n1)), 2),  # higher amounts
            "currency":              "USD",
            "txn_datetime":          timestamps,
            "hour_of_day":           rng.integers(0, 6, n1),   # late night
            "day_of_week":           [t.weekday() for t in timestamps],
            "is_weekend":            [(t.weekday() >= 5) * 1 for t in timestamps],
            "card_present":          np.zeros(n1, dtype=int),  # always CNP
            "is_emv_chip":           np.zeros(n1, dtype=int),
            "is_contactless":        np.zeros(n1, dtype=int),
            "is_3ds_authenticated":  rng.choice([0, 1], n1, p=[0.85, 0.15]),  # bypass 3DS
            "cardholder_country":    profiles["home_country"].values,
            "merchant_country":      rng.choice(list(self.HIGH_RISK_COUNTRIES), n1),
            "is_domestic":           np.zeros(n1, dtype=int),
            "card_age_days":         profiles["card_age_days"].values,
            "device_fingerprint_match": np.zeros(n1, dtype=int),   # new device
            "is_new_device":         np.ones(n1, dtype=int),
            "txn_count_1h":          rng.poisson(7, n1),   # high velocity
            "txn_count_6h":          rng.poisson(15, n1),
            "txn_count_24h":         rng.poisson(30, n1),
            "txn_count_7d":          rng.poisson(50, n1),
            "amount_sum_1h":         np.round(np.abs(rng.lognormal(6.5, 1.0, n1)), 2),
            "amount_sum_24h":        np.round(np.abs(rng.lognormal(7.5, 1.0, n1)), 2),
            "unique_merchants_24h":  rng.integers(5, 20, n1),
            "unique_countries_7d":   rng.integers(3, 8, n1),
            "declined_count_24h":    rng.poisson(4, n1),
            "days_since_last_txn":   rng.integers(0, 2, n1),
            "distance_from_home_km": np.abs(rng.exponential(5000, n1)),
            "impossible_travel_flag":rng.choice([0, 1], n1, p=[0.30, 0.70]),
            "ip_country_mismatch":   np.ones(n1, dtype=int),
            "is_high_risk_country":  np.ones(n1, dtype=int),
            "is_high_risk_mcc":      np.ones(n1, dtype=int),
            "merchant_base_fraud_rate": rng.uniform(0.02, 0.05, n1),
            "is_card_not_present":   np.ones(n1, dtype=int),
            "billing_address_change_flag": rng.choice([0, 1], n1, p=[0.40, 0.60]),
            "days_since_last_change": rng.integers(0, 5, n1),
            "is_recurring":          np.zeros(n1, dtype=int),
            "velocity_score":        np.clip(rng.beta(8, 2, n1), 0, 1),
            "is_fraud":              1,
        }))

        # ── Pattern 2: Account Takeover (ATO) ──
        n2 = pattern_counts["account_takeover"]
        ch_ids = rng.choice(self.cardholder_profiles.index, n2)
        mer_ids = rng.choice(self.merchant_profiles.index, n2)
        profiles = self.cardholder_profiles.loc[ch_ids].reset_index()
        merchants = self.merchant_profiles.loc[mer_ids].reset_index()
        offsets = rng.uniform(0, days * 86400, n2).astype(int)
        timestamps = [start + timedelta(seconds=int(o)) for o in offsets]

        frames.append(pd.DataFrame({
            "cardholder_id":         ch_ids,
            "merchant_id":           mer_ids,
            "merchant_category":     rng.choice(["retail", "online", "telecom"], n2, p=[0.40, 0.40, 0.20]),
            "mcc":                   rng.choice([5999, 5965, 4814], n2, p=[0.40, 0.40, 0.20]),
            "amount":                np.round(np.abs(rng.lognormal(6.0, 0.8, n2)), 2),
            "currency":              "USD",
            "txn_datetime":          timestamps,
            "hour_of_day":           rng.integers(2, 8, n2),
            "day_of_week":           [t.weekday() for t in timestamps],
            "is_weekend":            [(t.weekday() >= 5) * 1 for t in timestamps],
            "card_present":          rng.choice([0, 1], n2, p=[0.70, 0.30]),
            "is_emv_chip":           rng.choice([0, 1], n2, p=[0.60, 0.40]),
            "is_contactless":        rng.choice([0, 1], n2, p=[0.60, 0.40]),
            "is_3ds_authenticated":  rng.choice([0, 1], n2, p=[0.70, 0.30]),
            "cardholder_country":    profiles["home_country"].values,
            "merchant_country":      rng.choice(["US", "RO", "NG"], n2, p=[0.40, 0.30, 0.30]),
            "is_domestic":           rng.choice([0, 1], n2, p=[0.40, 0.60]),
            "card_age_days":         profiles["card_age_days"].values,
            "device_fingerprint_match": np.zeros(n2, dtype=int),  # different device → ATO signal
            "is_new_device":         np.ones(n2, dtype=int),
            "txn_count_1h":          rng.poisson(10, n2),  # burst pattern
            "txn_count_6h":          rng.poisson(20, n2),
            "txn_count_24h":         rng.poisson(40, n2),
            "txn_count_7d":          rng.poisson(45, n2),
            "amount_sum_1h":         np.round(np.abs(rng.lognormal(7.0, 0.8, n2)), 2),
            "amount_sum_24h":        np.round(np.abs(rng.lognormal(8.0, 0.8, n2)), 2),
            "unique_merchants_24h":  rng.integers(8, 25, n2),
            "unique_countries_7d":   rng.integers(2, 6, n2),
            "declined_count_24h":    rng.poisson(5, n2),
            "days_since_last_txn":   np.zeros(n2, dtype=int),  # rapid fire
            "distance_from_home_km": np.abs(rng.exponential(2000, n2)),
            "impossible_travel_flag":rng.choice([0, 1], n2, p=[0.20, 0.80]),
            "ip_country_mismatch":   rng.choice([0, 1], n2, p=[0.20, 0.80]),
            "is_high_risk_country":  rng.choice([0, 1], n2, p=[0.40, 0.60]),
            "is_high_risk_mcc":      rng.choice([0, 1], n2, p=[0.30, 0.70]),
            "merchant_base_fraud_rate": rng.uniform(0.01, 0.04, n2),
            "is_card_not_present":   rng.choice([0, 1], n2, p=[0.30, 0.70]),
            "billing_address_change_flag": rng.choice([0, 1], n2, p=[0.20, 0.80]),
            "days_since_last_change": rng.integers(0, 3, n2),  # recent change → ATO
            "is_recurring":          np.zeros(n2, dtype=int),
            "velocity_score":        np.clip(rng.beta(9, 1.5, n2), 0, 1),
            "is_fraud":              1,
        }))

        # ── Pattern 3: Lost/Stolen Card ──
        n3 = pattern_counts["lost_stolen"]
        ch_ids = rng.choice(self.cardholder_profiles.index, n3)
        mer_ids = rng.choice(self.merchant_profiles.index, n3)
        profiles = self.cardholder_profiles.loc[ch_ids].reset_index()
        merchants = self.merchant_profiles.loc[mer_ids].reset_index()
        offsets = rng.uniform(0, days * 86400, n3).astype(int)
        timestamps = [start + timedelta(seconds=int(o)) for o in offsets]

        frames.append(pd.DataFrame({
            "cardholder_id":         ch_ids,
            "merchant_id":           mer_ids,
            "merchant_category":     rng.choice(["grocery", "gas", "retail"], n3, p=[0.35, 0.35, 0.30]),
            "mcc":                   rng.choice([5411, 5541, 5999], n3, p=[0.35, 0.35, 0.30]),
            "amount":                np.round(np.abs(rng.uniform(5, 200, n3)), 2),  # small test txns
            "currency":              "USD",
            "txn_datetime":          timestamps,
            "hour_of_day":           rng.integers(18, 24, n3),  # evening
            "day_of_week":           [t.weekday() for t in timestamps],
            "is_weekend":            [(t.weekday() >= 5) * 1 for t in timestamps],
            "card_present":          np.ones(n3, dtype=int),  # physical card
            "is_emv_chip":           rng.choice([0, 1], n3, p=[0.50, 0.50]),
            "is_contactless":        rng.choice([0, 1], n3, p=[0.40, 0.60]),
            "is_3ds_authenticated":  rng.choice([0, 1], n3, p=[0.90, 0.10]),
            "cardholder_country":    profiles["home_country"].values,
            "merchant_country":      profiles["home_country"].values,  # domestic
            "is_domestic":           np.ones(n3, dtype=int),
            "card_age_days":         profiles["card_age_days"].values,
            "device_fingerprint_match": rng.choice([0, 1], n3, p=[0.50, 0.50]),
            "is_new_device":         rng.choice([0, 1], n3, p=[0.60, 0.40]),
            "txn_count_1h":          rng.poisson(4, n3),  # multiple small txns
            "txn_count_6h":          rng.poisson(8, n3),
            "txn_count_24h":         rng.poisson(12, n3),
            "txn_count_7d":          rng.poisson(20, n3),
            "amount_sum_1h":         np.round(rng.uniform(50, 500, n3), 2),
            "amount_sum_24h":        np.round(rng.uniform(100, 1000, n3), 2),
            "unique_merchants_24h":  rng.integers(3, 10, n3),
            "unique_countries_7d":   np.ones(n3, dtype=int),
            "declined_count_24h":    rng.poisson(2, n3),
            "days_since_last_txn":   np.zeros(n3, dtype=int),
            "distance_from_home_km": np.abs(rng.exponential(50, n3)),
            "impossible_travel_flag":np.zeros(n3, dtype=int),
            "ip_country_mismatch":   rng.choice([0, 1], n3, p=[0.80, 0.20]),
            "is_high_risk_country":  np.zeros(n3, dtype=int),
            "is_high_risk_mcc":      rng.choice([0, 1], n3, p=[0.60, 0.40]),
            "merchant_base_fraud_rate": rng.uniform(0.005, 0.02, n3),
            "is_card_not_present":   np.zeros(n3, dtype=int),
            "billing_address_change_flag": np.zeros(n3, dtype=int),
            "days_since_last_change": rng.integers(30, 200, n3),
            "is_recurring":          np.zeros(n3, dtype=int),
            "velocity_score":        np.clip(rng.beta(6, 3, n3), 0, 1),
            "is_fraud":              1,
        }))

        # ── Patterns 4+5: Synthetic Identity + Friendly Fraud ──
        n45 = n - n1 - n2 - n3
        ch_ids = rng.choice(self.cardholder_profiles.index, n45)
        mer_ids = rng.choice(self.merchant_profiles.index, n45)
        profiles = self.cardholder_profiles.loc[ch_ids].reset_index()
        merchants = self.merchant_profiles.loc[mer_ids].reset_index()
        offsets = rng.uniform(0, days * 86400, n45).astype(int)
        timestamps = [start + timedelta(seconds=int(o)) for o in offsets]

        frames.append(pd.DataFrame({
            "cardholder_id":         ch_ids,
            "merchant_id":           mer_ids,
            "merchant_category":     merchants["category"].values,
            "mcc":                   merchants["mcc"].values,
            "amount":                np.round(np.abs(rng.lognormal(5.0, 1.5, n45)), 2),
            "currency":              "USD",
            "txn_datetime":          timestamps,
            "hour_of_day":           rng.integers(0, 24, n45),
            "day_of_week":           [t.weekday() for t in timestamps],
            "is_weekend":            [(t.weekday() >= 5) * 1 for t in timestamps],
            "card_present":          rng.choice([0, 1], n45, p=[0.60, 0.40]),
            "is_emv_chip":           rng.choice([0, 1], n45, p=[0.40, 0.60]),
            "is_contactless":        rng.choice([0, 1], n45, p=[0.60, 0.40]),
            "is_3ds_authenticated":  rng.choice([0, 1], n45, p=[0.50, 0.50]),
            "cardholder_country":    profiles["home_country"].values,
            "merchant_country":      merchants["country"].values,
            "is_domestic":           (profiles["home_country"].values == merchants["country"].values).astype(int),
            "card_age_days":         rng.integers(1, 30, n45),  # new card = synthetic ID signal
            "device_fingerprint_match": rng.choice([0, 1], n45, p=[0.40, 0.60]),
            "is_new_device":         rng.choice([0, 1], n45, p=[0.30, 0.70]),
            "txn_count_1h":          rng.poisson(3, n45),
            "txn_count_6h":          rng.poisson(6, n45),
            "txn_count_24h":         rng.poisson(15, n45),
            "txn_count_7d":          rng.poisson(30, n45),
            "amount_sum_1h":         np.round(np.abs(rng.lognormal(5.0, 1.5, n45)), 2),
            "amount_sum_24h":        np.round(np.abs(rng.lognormal(6.5, 1.5, n45)), 2),
            "unique_merchants_24h":  rng.integers(2, 12, n45),
            "unique_countries_7d":   rng.integers(1, 4, n45),
            "declined_count_24h":    rng.poisson(2, n45),
            "days_since_last_txn":   rng.integers(0, 5, n45),
            "distance_from_home_km": np.abs(rng.exponential(300, n45)),
            "impossible_travel_flag":rng.choice([0, 1], n45, p=[0.60, 0.40]),
            "ip_country_mismatch":   rng.choice([0, 1], n45, p=[0.50, 0.50]),
            "is_high_risk_country":  rng.choice([0, 1], n45, p=[0.50, 0.50]),
            "is_high_risk_mcc":      rng.choice([0, 1], n45, p=[0.40, 0.60]),
            "merchant_base_fraud_rate": rng.uniform(0.01, 0.05, n45),
            "is_card_not_present":   rng.choice([0, 1], n45, p=[0.30, 0.70]),
            "billing_address_change_flag": rng.choice([0, 1], n45, p=[0.50, 0.50]),
            "days_since_last_change": rng.integers(0, 10, n45),
            "is_recurring":          rng.choice([0, 1], n45, p=[0.80, 0.20]),
            "velocity_score":        np.clip(rng.beta(7, 2, n45), 0, 1),
            "is_fraud":              1,
        }))

        return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# SECTION 2: Feature Engineering Pipeline
# ──────────────────────────────────────────────────────────────
class FeatureEngineeringPipeline:
    """
    Transforms raw transaction data into ML-ready features.

    Mirrors Visa's internal Feature Store design:
    - Real-time features: computed at inference time
    - Batch features: pre-computed daily from Hive
    - Entity features: cardholder/merchant profile features
    """

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.feature_stats = {}   # stores mean/std for z-score
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """Fit on training data and transform."""
        df = df.copy()
        df = self._compute_derived_features(df, fit=True)
        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_cols
        self.is_fitted = True
        logger.info(f"Feature matrix: {X_scaled.shape[0]:,} × {X_scaled.shape[1]} features")
        return X_scaled, df["is_fraud"].values, feature_cols

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        df = df.copy()
        df = self._compute_derived_features(df, fit=False)
        X = df[self.feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        return self.scaler.transform(X)

    def _compute_derived_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Engineer features from raw fields."""

        # ── Amount features ──
        df["amount_log1p"] = np.log1p(df["amount"])
        mean_amt = df["amount"].mean() if fit else self.feature_stats.get("amount_mean", df["amount"].mean())
        std_amt  = df["amount"].std()  if fit else self.feature_stats.get("amount_std", df["amount"].std())
        if fit:
            self.feature_stats["amount_mean"] = mean_amt
            self.feature_stats["amount_std"]  = std_amt
        df["amount_zscore"]      = (df["amount"] - mean_amt) / (std_amt + 1e-9)
        df["amount_vs_merchant_avg"] = df["amount"] / (df.get("merchant_base_fraud_rate", 0.01) * 10000 + 1)
        df["amount_log1p_sq"]    = df["amount_log1p"] ** 2

        # ── Time features ──
        df["is_night"]           = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 4)).astype(int)
        df["is_business_hours"]  = ((df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17)).astype(int)
        df["hour_sin"]           = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"]           = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["dow_sin"]            = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]            = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # ── Velocity ratio features ──
        df["vel_ratio_1h_24h"]   = df["txn_count_1h"] / (df["txn_count_24h"] + 1)
        df["vel_ratio_6h_7d"]    = df["txn_count_6h"] / (df["txn_count_7d"] + 1)
        df["amount_vel_ratio"]   = df["amount"] / (df["amount_sum_24h"] + 1)

        # ── Risk composite scores ──
        df["card_security_score"] = (
            df["is_emv_chip"] * 0.30
            + df["is_3ds_authenticated"] * 0.35
            + df["device_fingerprint_match"] * 0.20
            + (1 - df["is_new_device"]) * 0.15
        )
        df["geographic_risk_score"] = (
            df["is_high_risk_country"] * 0.35
            + df["impossible_travel_flag"] * 0.30
            + df["ip_country_mismatch"] * 0.20
            + (1 - df["is_domestic"]) * 0.15
        )
        df["velocity_composite"] = (
            df["txn_count_1h"] * 0.4
            + df["txn_count_6h"] * 0.3
            + df["declined_count_24h"] * 2.0
            + df["unique_merchants_24h"] * 0.3
        )
        df["channel_risk"]       = (
            (1 - df["card_present"]) * 0.40
            + df["is_card_not_present"] * 0.30
            + (1 - df["is_3ds_authenticated"]) * 0.20
            + df["billing_address_change_flag"] * 0.10
        )

        # ── Interaction features ──
        df["high_amount_foreign"]  = df["amount_log1p"] * (1 - df["is_domestic"])
        df["night_cnp"]            = df["is_night"] * (1 - df["card_present"])
        df["velocity_x_foreign"]   = df["velocity_composite"] * (1 - df["is_domestic"])
        df["new_device_high_amt"]  = df["is_new_device"] * df["amount_log1p"]
        df["atm_declined_velocity"]= df["declined_count_24h"] * df["txn_count_1h"]

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Return the final list of ML features (exclude raw IDs, labels, etc.)."""
        exclude = {
            "txn_id", "cardholder_id", "merchant_id", "txn_datetime",
            "currency", "is_fraud", "merchant_category", "cardholder_country",
            "merchant_country", "merchant_base_fraud_rate",
        }
        return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
