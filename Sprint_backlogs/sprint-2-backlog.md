## Sprint planing 

## Participants 
- developer 1 : @coz1686 (Piotr Ciupiński)
- developer 2:  @cgy3920 (Veronika Anokhina)
- developer 3: @cuu7703 (Rafał Mironko)

### Sprint Goal  
Prepare the data for training and testing the model. *(Subtasks from ST01-ST04 finished)*  

## Product backlog items
- Data distribution analysed (remaining ST01)
- Data cleaned (ST02)
- Trips labeled (ST04)
- Tools for model comparison selected (remaining ST03)
- Anomalies defined (ST04)
- All processes documented (ST01, ST02, ST03, ST04)
- Product repository cleaned and organized


### planned activities:
  - Analyze data distribution and document it
  - Define abnormal trajectories and document it
  - Fill missing values in the data
  - Clean the data from errors
  - Clean the data from unnecessary attributes
  - Sort obvious noise from the data
  - Label trips as normal or abnormal
  - Document the labeling process and reasoning
  - Select tools for models result analysis

---

## Work Organization (Sprint Plan)


@coz1686 (Piotr Ciupiński)
@cgy3920 (Veronika Anokhina)
@cuu7703 (Rafał Mironko)


### Week 1: Foundation & Initial Processing 

| Task                                       | Description                                                            | Responsible                                          | Responsibility Change | Estimated Time | Time Spent | Dependency           |
|--------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------|-----------------------|----------------|------------|----------------------|
| **1. Analyze data distributions - {ST01}** | Statistical summaries, visualizations, documentation                   | @cuu7703 (Rafał Mironko)                             | -                     | 6h (2h+2h+2h)  | -          | None (parallel)      |
| **2. Draft anomaly definitions - {ST04}**  | Research anomalies, define criteria, document, start labeling research | @cuu7703 (Rafał Mironko), @coz1686 (Piotr Ciupiński) | -                     | 6h (2h+2h+2h)  | -          | ST01 (partial)       |
| **3. Group Session**                       | Consensus on anomalies + review data analysis                          | All                                                  | -                     | 1.5h           | 1.5h       | ST01                 |
| **4. Develop cleaning pipeline - {ST02}**  | Missing value handling, error removal, drop attributes                 | @cgy3920 (Veronika Anokhina)                         | -                     | 9h (4h+3h+2h)  | 10h        | ST01 complete & 2, 3 |
| **5. Research analysis tools - {ST03}**    | Compare tools for model evaluation, data labeling                      | @coz1686 (Piotr Ciupiński)                           | -                     | 3h             | -          | None (parallel)      |

**Whole week time estimate: 25.5h**\
**Actual time spend:**
---

### Week 2: Refinement & Labeling 

| Task                                    | Description                                                 | Responsible                  | Responsibility Change                                | Estimated Time | Time Spent | Dependency                                |
|-----------------------------------------|-------------------------------------------------------------|------------------------------|------------------------------------------------------|----------------|------------|-------------------------------------------|
| **1. Implement noise removal - {ST02}** | Remove noise data, play with the threshold of what is noise | @coz1686 (Piotr Ciupiński)   | @cgy3920 (Veronika Anokhina)                         | 6h (3h+3h)     | 8h         | Week 1 ST02 complete                      |
| **2. Label sample trips - {ST04}**      | Apply definitions, revise labels, document process          | Everyone                     | @cuu7703 (Rafał Mironko), @coz1686 (Piotr Ciupiński) | 10h (5h+3h+2h) | -          | Week 1 ST04 finalized + ST02 cleaned data |
| **3. Integrate tools - {ST03}**         | Docker setup for selected tools                             | @coz1686 (Piotr Ciupiński)   | -                                                    | 3h             | -          | Week 1 ST03 research done                 |
| **4. Cleanup repository**               | Organize code/docs, final checks                            | @cgy3920 (Veronika Anokhina) | Everyone                                             | 1h             | -          | All other tasks complete                  |
| **5. Buffer/Contingency**               | Revisions, meetings, overflows                              | Flexible                     | -                                                    | 10h            | -          | -                                         |

**Flexible hours spent:**

| Person                       | Task                                                   | Time Spent |
|------------------------------|--------------------------------------------------------|------------|
| @cgy3920 (Veronika Anokhina) | Research for data cleaning, research for data labeling | 3h         |
| @cuu7703 (Rafał Mironko)     |                                                        |            |
| @coz1686 (Piotr Ciupiński)   |                                                        |            |



**Whole week time estimate: 30h**\
**Actual time spend:**
---

### Product Increments (After Sprint)


### Sprint Review / Retrospective (After Sprint)

