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
 
## Results (test set, 50 photos)
 
| model | test accuracy | test macro-F1 |
|---|---|---|
| Majority-class baseline | 0.76 | 0.288 |
| Color histograms + logistic regression | 0.64 | 0.527 |
| MobileNetV2 features (frozen) + logistic regression | 1.00 | 1.000 |
| Small CNN trained from scratch | 0.80 | 0.494 |
 
**Main findings**
- Accuracy alone is misleading: always guessing `exploration` gets 76% but macro-F1 is only 0.288.
- The CNN never predicted `work_study` (0 of 6 correct on test).
- MobileNetV2 features scored perfectly, but the test set is only 50 photos and I took about 60 work/study photos on a single day, so I checked for leakage and ran grouped cross-validation (see Section 9b of the notebook).
- The CNN's validation curves reflect a split problem, not classic overfitting: the validation set is 86% `work_study` while training has only 32 such photos.
## Pipeline

```mermaid
flowchart TB
  A["Raw data<br/>personal phone photos"] --> B["Manual labeling by rules<br/>work_study: laptop with working tab<br/>exploration: mostly scenery<br/>social: 3+ people or food table"]
  B --> C["Google Drive<br/>3 class folders<br/>nature merged into exploration"]
  C --> D["Preprocessing<br/>read capture date: EXIF, filename fallback<br/>group photos into 2-hour sessions"]
  D --> E["Dataframe<br/>path, label, date, session"]
  E --> F["Stratified grouped split<br/>whole sessions stay together"]
  F --> G{"All 3 classes in<br/>train, val and test?"}
  G -- "No: new random seed" --> F
  G -- Yes --> TR["Train X, Train y<br/>about 70%"]
  G -- Yes --> VA["Val X, Val y<br/>about 15%"]
  G -- Yes --> TE["Test X, Test y<br/>about 15%, reported once"]

  TR & VA & TE --> P["Image preprocessing<br/>fix rotation, convert to RGB<br/>resize 224x224, cache locally"]

  P --> F1["Color histograms<br/>plus channel mean and std"]
  P --> F2["MobileNetV2 embeddings<br/>1280 features per photo"]
  P --> F3["Raw pixels<br/>augmentation on train only"]
  IW["ImageNet pretrained weights<br/>frozen, not trained"] --> F2

  MS["Model selection<br/>logistic regression: C chosen on validation set<br/>CNN: early stopping on validation loss"]

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

  ER --> PM["Performance metrics<br/>accuracy, macro-F1,<br/>grouped 5-fold CV mean and std,<br/>95% interval"]
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
  class F,G step
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
 
