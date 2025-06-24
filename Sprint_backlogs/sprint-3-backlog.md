## Sprint planing

## Participants
- developer 1 : @coz1686 (Piotr Ciupiński)
- developer 2:  @cgy3920 (Veronika Anokhina)
- developer 3: @cuu7703 (Rafał Mironko)

### Sprint Goal
Finish unsupervised labeling of the data and create training pipeline for chosen models and LSTM model.\
*(Subtasks from ST04 finished)* \
*(Subtasks from ST05-ST06 building whole pipeline)*
 
## Product backlog items
- Trips labeled by unsupervised ML (ST04)
- Revise previously made choice of models (ST05)
- Create pipeline for models training (ST05)
- Implement training pipeline LSTM model (ST05)

### planned activities:
1. Data Labeling & Validation
   - Apply clustering/auto-labeling tools to remaining unlabeled data.
   - Compare accuracy of unsupervised labels against manual labels
2. Model Preparation
   - Re-evaluate algorithms based on cleaned data and label quality
   - Prepare data for each algorithm (e.g., feature engineering)
   - Create training pipeline for at least 3 models with different algorithms
3. LSTM Implementation1
   - Research/adapt architecture for trajectory anomalies.
   - Implement training pipeline LSTM model
   
- Document the process and results

---

## Work Organization (Sprint Plan)

### Week 1:

| Task                                                                            | Description                                                    | Responsible                                           | Responsible Change                                     | Estimated Time | Time Spent | Dependency |
|---------------------------------------------------------------------------------|----------------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------|----------------|------------|------------|
| **1. Apply clustering/auto-labeling tools to remaining unlabeled data.** (ST04) |                                                                | @cgy3920 (Veronika Anokhina)                          | Dropped to focus more on LSTM                          | 6h             | 3h         | None       |
| **2. Compare accuracy of unsupervised labels against manual labels** (ST04)     |                                                                | @cgy3920 (Veronika Anokhina)                          | Task change: fully connecting cleaning pipeline {ST09} | 2h             | 3h         | Task 1     |
| **3. Revise models choice** (ST05)                                              | Re-evaluate algorithms based on cleaned data and label quality | All                                                   |                                                        | 2h             | 1h         | Task 2     |
| **Review meeting**                                                              |                                                                | All                                                   |                                                        | 1h             | 1h         |            |
| **4. Data Preparation** (ST05)                                                  | Adapt cleaned/labeled data for each algorithm                  | @cgy3920 (Veronika Anokhina)                          | @coz1686 (Piotr Ciupiński) & @cuu7703 (Rafał Mironko)  | 4h             | 4h         | Task 3     |
| **5. Implement pipeline Model 1** (ST05)                                        | Code + basic parameters train                                  | @coz1686 (Piotr Ciupiński) & @cuu7703 (Rafał Mironko) |                                                        | 8h             | 6h         | Task 4     |
| **6. Implement pipeline Model 2** (ST05)                                        | Code + basic parameters train                                  | @cuu7703 (Rafał Mironko)                              |                                                        | 3h             | 6h         | Task 4     |
| **Meetings**                                                                    |                                                                | All                                                   |                                                        | 1h+1.5h        | 1h         |            |


**Whole week time estimate:28.5h**\
**Actual time spend: 26.h**

### Week 2:

| Task                                                   | Description                                                    | Responsible                  | Responsibility Change                    | Estimated Time | Time Spent | Dependency    |
|--------------------------------------------------------|----------------------------------------------------------------|------------------------------|------------------------------------------|----------------|------------|---------------|
| **0. Continue pipeline implementation Model 2** (ST05) | Code + basic parameters train                                  | @cuu7703 (Rafał Mironko)     |                                          | 5h             | 6h         | Week 1 Task 4 |
| **1. Implement pipeline Model 3** (ST05)               | Code + basic parameters train                                  | @coz1686 (Piotr Ciupiński)   | Additional model pipeline implementation | 10h            | 10h        | Week 1 Task 4 |
| **2. Research LSTM**(ST05)                             | Research/adapt architecture for trajectory anomalies           | @cgy3920 (Veronika Anokhina) |                                          | 4h             | 4h         | Week 1 Task 2 |
| **3. Implement LSTM** (ST06)                           | Write the LSTM model code, integrate with cleaned/labeled data | @cgy3920 (Veronika Anokhina) |                                          | 10h            | 10h        | Task 2        |
| **4. Documentation & Integration** (ST09)              | Organize code/docs, final checks                               | All                          |                                          | 2h             | 1h         |               |
| **Meetings**                                           |                                                                | All                          |                                          | 2h             | 1.5h       |               |


**Whole week time estimate:33h**\

**Actual time spend:33h**

**Time per person:**

| Person                       | Time Spent |
|------------------------------|------------|
| @cgy3920 (Veronika Anokhina) | 25.5       |
| @coz1686 (Piotr Ciupiński)   | 25.5       |
| @cuu7703 (Rafał Mironko)     | 27.5       |

### Product Increments (After Sprint)

**Apply clustering/auto-labeling tools to remaining unlabeled data (DROPPED) - {ST04}** (branch _data-model_preparation_): 
- `labeling/*`

**Fully connecting cleaning pipeline - {ST09}** (branch _data-cleaning_): 
- `data-cleaning/*.py`

**Revise models choice - {ST05}**  
-`models\README.md`

**Data Preparation - {ST05}**    
- incorporated in `models\{model_training_pipelines}.py` & `models\data_visualizer.py` 
- `models\model_features.md`

**Implement pipeline Models - {ST05}**       
- `models\iso_for.py`
- `models\log_reg.py`
- `models\oc_svm.py`
- `models\random_for.py`

**Research LSTM - {ST05}**  
   - `models\LSTM\LSTM_explanation.md`
-  _in our heads_

**Implement LSTM - {ST05}**  
- `models\LSTM\*.py`
- `models\LSTM\*.ipynb`

**Documentation & Integration- {ST09}** - 
- gitlab directory 

### Sprint Review / Retrospective (After Sprint)
