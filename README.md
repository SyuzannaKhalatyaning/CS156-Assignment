# What is my camera roll made of? Classifying personal photos by activity
 
CS156 Pipeline Assignment 1
 
As a Minerva student my life splits into studying, working, socializing and exploring each rotation city, and sometimes I can't tell which one I'm doing. My phone holds 12,365 photos, so I built a machine learning pipeline that sorts photos into three classes and compared four different approaches.
 
**Final notebook (with all outputs): [`CS156_First_pipeline_clean.ipynb`](CS156_First_pipeline_clean.ipynb)**
Earlier drafts are kept in [`archive/`](archive/).
 
## Classes
 
| class | how I labeled it |
|---|---|
| `work_study` | my laptop screen is visible and shows a tab proving I was working or studying (not a movie) |
| `social` | 3+ people, or a table with more food than one person could eat, or a crowd/event |
| `exploration` | scenery fills at least ~60% of the frame |
 
`social` and `exploration` overlap on purpose (some scenery photos include friends). I wanted to see whether a model could separate them even though I sometimes can't.
 
## Results

Test set (50 photos, single split) and grouped 5-fold cross-validation (mean ± sd over 5 session-grouped splits):

| model | test accuracy | test macro-F1 | CV accuracy | CV macro-F1 |
|---|---|---|---|---|
| Majority-class baseline | 0.76 | 0.288 | n/a | n/a |
| Color histograms + logistic regression | 0.64 | 0.527 | 0.68 ± 0.06 | 0.56 ± 0.06 |
| MobileNetV2 features (frozen) + logistic regression | 1.00 | 1.000 | 0.93 ± 0.05 | 0.89 ± 0.09 |
| Small CNN trained from scratch | 0.80 | 0.494 | n/a | n/a |

The CV was run only for the two logistic-regression models. The 1.00 on the single test split was optimistic, and the CV scores are the more reliable estimate.
 
**Main findings**
- Accuracy alone is misleading: always guessing `exploration` gets 76% but macro-F1 is only 0.288.
- The CNN never predicted `work_study` (0 of 6 correct on test).
- MobileNetV2 features scored 1.00 on the single 50-photo test split, but that was optimistic. Grouped 5-fold CV gives 0.93 ± 0.05 accuracy and 0.89 ± 0.09 macro-F1. There were no near-duplicates between train and test (0 of 50 above 0.90 similarity), but about half of the test photos share a day with training photos, which can inflate scores.
- Even so, pretrained features (about 0.93) clearly beat color features (about 0.68), so they do separate my classes.
- The CNN's validation curves reflect a split problem, not classic overfitting: the validation set is 86% `work_study` while training has only 32 such photos.
  
## Pipeline

```mermaid
flowchart TB
  A["Raw data<br/>personal phone photos"] --> B["Manual labeling by rules<br/>work_study: laptop with working tab<br/>exploration: mostly scenery<br/>social: 3+ people or food table"]
  B --> C["Google Drive<br/>3 class folders<br/>nature merged into exploration"]
  C --> D["Preprocessing<br/>read capture date: EXIF, filename fallback<br/>group photos into 15-minute sessions"]
  D --> E["Dataframe<br/>path, label, date, session"]
  E --> F["Stratified grouped split<br/>whole sessions stay together"]
  F --> TR["Train X, Train y<br/>about 54%"]
  F --> VA["Val X, Val y<br/>about 36%"]
  F --> TE["Test X, Test y<br/>about 10%, reported once"]
  TR & VA & TE --> P["Image preprocessing<br/>fix rotation, convert to RGB<br/>resize 224x224, cache locally"]

  P --> F1["Color histograms<br/>plus channel mean and std"]
  P --> F2["MobileNetV2 embeddings<br/>1280 features per photo"]
  P --> F3["Raw pixels<br/>augmentation on train only"]
  IW["ImageNet pretrained weights<br/>frozen, not trained"] --> F2
  MS["Settings<br/>logistic regression: C = 0.1 fixed<br/>CNN: early stopping on validation loss"]

  subgraph FIT["Fit, predict, measure"]
    BL["Majority-class baseline"]
    M1["Untrained logistic regression<br/>on color features"]
    M2["Untrained logistic regression<br/>on MobileNetV2 features"]
    M3["Untrained small CNN"]
    BL --> T0["Trained baseline"]
    M1 -- "fit parameters on train" --> T1["Trained model"]
    M2 -- "fit parameters on train" --> T2["Trained model"]
    M3 -- "fit parameters on train" --> T3["Trained CNN"]
    T0 & T1 & T2 & T3 -- "make predictions on test" --> PR["Predictions"]
    PR -- "compare with Test y" --> ER["Errors<br/>confusion matrices, per-class precision and recall,<br/>inspect misclassified photos"]
  end

  F1 --> M1
  F2 --> M2
  F3 --> M3
  MS --> M1
  MS --> M2
  MS --> M3

  ER --> PM["Performance metrics<br/>accuracy, macro-F1,<br/>grouped 5-fold CV mean and std"]
  ER --> VZ["Visualizations<br/>learning curves, confusion matrices,<br/>model comparison table"]

  classDef data fill:#F4A460,stroke:#555,color:#000
  classDef model fill:#A9C1F0,stroke:#555,color:#000
  classDef pred fill:#B7A6D4,stroke:#555,color:#000
  classDef out fill:#77F24F,stroke:#555,color:#000
  classDef metric fill:#F2F25E,stroke:#555,color:#000
  classDef step fill:#F8DC5A,stroke:#555,color:#000
  class A,B,C,D,E,TR,VA,TE,P,F1,F2,F3 data
  class IW,MS,BL,M1,M2,M3,T0,T1,T2,T3 model
  class PR pred
  class ER,VZ out
  class PM metric
  class F step
```
## Data and privacy
 
The photos are personal and are **not** included in this repository (my Google Drive is private). The notebook's saved outputs show every result, so you can read it without re-running it. To re-run it you would need your own photos in `MyDrive/data/Work`, `Socials` and `Exploration`.
 
## How to run
 
1. Open the notebook in Google Colab.
2. Put your labeled photos on Drive in the three folders above.
3. Runtime → Run all (the first cell mounts Drive and installs `pillow-heif` for `.heic` photos).

Libraries: TensorFlow/Keras, scikit-learn, pandas, NumPy, Pillow, matplotlib.
 
## References
 
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, 4510-4520.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825-2830.
- TensorFlow tutorials: [Image classification](https://www.tensorflow.org/tutorials/images/classification), [Transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- Full list in the notebook.
 
