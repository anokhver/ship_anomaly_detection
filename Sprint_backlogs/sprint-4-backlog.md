## Sprint planing

## Participants
- developer 1 : @coz1686 (Piotr Ciupiński)
- developer 2:  @cgy3920 (Veronika Anokhina)
- developer 3: @cuu7703 (Rafał Mironko)

### Sprint Goal
Tune chosen models and LSTM model.
Choose the metrics for model evaluation & compare them.\
Build GUI.
Connect the whole pipeline for easy usage.\
Create final documentation.\

*(Subtasks from ST05-ST06 finished)*
*(Subtasks from ST07-ST09 finished)*
 

## Product backlog items
-final trained models (ST05-ST06)
-models comparison and analysis (ST07)
-frontend and backend GUI (ST08)
-Documentation (ST09)

### planned activities:
1. Model tuning
    - Tune parameters of models based on previous results.
2. Model evaluation
    - Choose metrics for model evaluation.
    - Compare models based on chosen metrics.
3. GUI Development
    - Finish frontend and backend GUI.
    - Integrate a cleaning pipeline and the application.
    - Create a script for easy usage of the application.
4. Create final documentation for the project.
    
---

## Work Organization (Sprint Plan)

### Week 1:

| Task                                               | Description | Responsible                  | Responsible Change                                                                    | Estimated Time | Time Spent     | Dependency |
|----------------------------------------------------|-------------|------------------------------|---------------------------------------------------------------------------------------|----------------|----------------|------------|
| **1. Tune models** (ST05-ST06)                     |             | @cgy3920 (Veronika Anokhina) | @coz1686 (Piotr Ciupiński) & @cuu7703 (Rafał Mironko)  & @cgy3920 (Veronika Anokhina) | 8h             | 10h + 4h + 10h | None       |
| **2. Adding visualization for LSTM** (ST08)        |             | @coz1686 (Piotr Ciupiński)   | @coz1686 (Piotr Ciupiński) & @cgy3920 (Veronika Anokhina)                             | 2h             | 1h + 1h        | None       |
| **3. Choose metrics for model evaluation.** (ST07) |             | All                          |                                                                                       | 2h             | 2h             | None       |
| **4. Creating backend for GUI** (ST08)             |             | @coz1686 (Piotr Ciupiński)   |                                                                                       | 8h             | 7h             | None       |
| **5. Creating GUI** (ST08)                         |             | @cuu7703 (Rafał Mironko)     |                                                                                       | 22h            | 22h            | None       |
| **5. Buffer & Meetings**                           |             | All                          |                                                                                       | 4h             | 4h             | None       |


**Whole week time estimate:46h**\

**Actual time spend:61h**

### Week 2:

| Task                                                         | Description                                  | Responsible                                               | Responsible Change                                        | Estimated Time | Time Spent   | Dependency |
|--------------------------------------------------------------|----------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------|----------------|--------------|------------|
| **1. Writing final model comparison** (ST07)                 |                                              | @cgy3920 (Veronika Anokhina)                              | @coz1686 (Piotr Ciupiński) & @cgy3920 (Veronika Anokhina) | 1.5h           | 3h + 1.5h    | Week 1     |
| **2. Integrate a cleaning pipeline into application** (ST08) |                                              | @coz1686 (Piotr Ciupiński) & @cgy3920 (Veronika Anokhina) |                                                           | 4h             | 3h           | Week 1     |
| **3. GUI data uploading integration** (ST08)                 |                                              | @cuu7703 (Rafał Mironko)                                  |                                                           | 2h             | 2h           | Task 2     |
| **4. Creating script for application run** (ST08)            |                                              | @coz1686 (Piotr Ciupiński)                                |                                                           | 1.5h           | 1.5h         | Task 3     |
| **5. Testing**                                               |                                              | All                                                       |                                                           | 4h             | 7h + 1h + 2h | All        |
| **6. Finish Documentation**                                  |                                              | All                                                       |                                                           | 2h             | 2h           | All        |
| **6. Buffer & Meetings**                                     | Meeting and fixing bugs found during testing | All                                                       |                                                           | 10h            | 10h(LSTM)    | Task 5     |


**Whole week time estimate:25h**\

**Actual time spent:33h**

**Time per person:**

| Person                       | Time Spent |
|------------------------------|------------|
| @cgy3920 (Veronika Anokhina) | 35.5h      |
| @coz1686 (Piotr Ciupiński)   | 40h        |
| @cuu7703 (Rafał Mironko)     | 37h        |

### Product Increments (After Sprint)


**Model evaluation**: 
- `models/model_tuning.md*`

**Model tuning**
- `models/model_tuning.md*`
- `models/{model_name}.py*`
- `models/LSTM/final_training.ipynb*`

**GUI**
- `frontend/*`  

**Backend**
- `backend/*`

**Integrated cleaning pipeline**:
- `backend/api/data_cleaning/*`

**Documentation**
- `README.md` thrue all project 
 
### Sprint Review / Retrospective (After Sprint)
