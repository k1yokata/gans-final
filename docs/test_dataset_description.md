# Test Procurement Dataset Description

## Purpose

This test dataset was created for the GAN-based public procurement risk detection project. It is used as a lightweight substitute for a real public procurement dataset during early development.

The dataset is connected with the dissertation topic:

**Machine Learning Based Early Warning for Corruption Risk in Public Procurement**

## Why a Test Dataset Is Used

Large public procurement datasets can be heavy and may overload a local computer. Therefore, the project first uses a small synthetic procurement dataset. This allows development, testing, preprocessing, and GAN model design without high local resource usage.

Later, the same pipeline can be adapted to real procurement data such as ProZorro or other open public procurement datasets.

## Number of Records

The dataset contains 5,000 synthetic tender records.

## Columns

| Column | Description |
|---|---|
| tender_id | Unique tender identifier |
| buyer_id | Buyer organization identifier |
| supplier_id | Supplier organization identifier |
| region | Procurement region |
| procedure_type | Type of procurement procedure |
| expected_value | Initial expected tender value |
| final_value | Final contract value |
| price_reduction_percent | Percentage reduction between expected and final value |
| number_of_bidders | Number of participating bidders |
| supplier_previous_wins | Number of previous wins by the supplier |
| buyer_previous_tenders | Number of previous tenders by the buyer |
| contract_days | Contract duration in days |
| risk_score | Rule-based corruption risk score from 0 to 100 |
| risk_label | Binary risk label: 1 means high risk, 0 means low or medium risk |

## Risk Score Logic

The risk score is calculated using simple rule-based indicators:

1. Single-bidder tender increases risk.
2. Direct award or negotiation procedure increases risk.
3. Very low price reduction increases risk.
4. Supplier with many previous wins increases risk.
5. Very high tender value increases risk.
6. Very short contract duration slightly increases risk.

## Role in the GAN Project

This dataset is used to train the first GAN prototype. The GAN model will learn the structure of procurement records and generate new synthetic procurement-like records.

The generated records will later be compared with the original test dataset using descriptive statistics and visualizations.
