# Census Income Analysis

Income prediction and customer segmentation using 1994-1995 US Census data.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib

Install:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Files

- `classification_model.py` - Random Forest model for income prediction
- `segmentation_model.py` - K-Means clustering for customer segmentation
- `census-bureau.data` - Dataset (not included in this repo)
- `census-bureau.columns` - Column names (not included in this repo)

## Running the Code

Make sure data files are in the same directory, then:

**Classification Model:**
```bash
python classification_model.py
```

Expected output: Model accuracy, confusion matrix, feature importance

**Segmentation Model:**
```bash
python segmentation_model.py
```

Expected output: Cluster profiles, elbow curve, PCA visualization

Runtime: ~2-3 minutes each

## Output Files

- `elbow_curve.png` - Optimal cluster selection
- `segmentation_viz.png` - Customer segments visualization

## Notes

Classification accuracy should be around 85-94%. 

Higher accuracy may indicate data leakage.

Segmentation uses k=3 clusters based on elbow method.
