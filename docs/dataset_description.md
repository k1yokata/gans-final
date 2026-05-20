# Dataset Description

## Dataset Topic

The project uses public procurement data for corruption risk analysis. The selected data domain is public procurement, because it is directly connected with the dissertation topic:

**Machine Learning Based Early Warning for Corruption Risk in Public Procurement**

## Dataset Source

The project is planned to use open public procurement data, such as ProZorro procurement datasets or similar open procurement datasets.

## Why This Dataset Is Relevant

Public procurement data contains information about tenders, buyers, suppliers, bids, prices, competition, and procurement procedures. These features can be used to analyze corruption risk indicators.

The dataset is suitable for this GAN project because it contains structured tabular records. GAN models can be trained to generate synthetic tabular procurement records that preserve statistical patterns of real procurement data.

## Main Data Entities

The dataset may include the following entities:

1. Tenders — procurement procedures and contract information.
2. Buyers — organizations that purchase goods or services.
3. Suppliers — companies or individuals that provide goods or services.
4. Bids — offers submitted by suppliers.
5. Procurement values — expected and final tender prices.
6. Dates — announcement, award, contract and completion dates.
7. Competition indicators — number of bidders and participants.

## Risk-Related Features

The following features may be useful for corruption risk detection:

1. Tender value.
2. Number of bidders.
3. Number of bids.
4. Supplier frequency.
5. Buyer frequency.
6. Difference between expected value and final value.
7. Procurement procedure type.
8. Single-bidder tenders.
9. Repeated supplier-buyer relationships.
10. Abnormally high or low procurement values.

## Data Load Reduction Strategy

The original dataset can be large. To reduce load on the local computer:

1. Raw CSV files will be stored on the external SSD.
2. Large files will not be pushed to GitHub.
3. Only small samples will be created locally.
4. Heavy preprocessing and GAN training will be done in Google Colab.
5. The local machine will be used mainly for code editing and Git.

## Planned Process

1. Place raw CSV files into `data/raw/`.
2. Read only selected columns from the raw dataset.
3. Create a small sample with limited number of rows.
4. Save the sample into `data/processed/`.
5. Use the processed sample for GAN model development.
