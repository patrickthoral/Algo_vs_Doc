# CDM-lite stay contract

The private hospital ETL writes one icu_stays row for each study-cohort ICU stay and for each
subsequent ICU stay required to characterize an observed seven-day readmission. Both groups use
the same schema.

## Cohort and follow-up flags

- is_study_cohort: True for stays belonging to the study cohort. False for follow-up context stays
  retained only to characterize a readmission. Context stays are enriched so their admission type
  and condition flags can be linked to the preceding study discharge. Notebook 1 removes them
  before writing the analysis dataset; they therefore do not enter enrolment counts, descriptive
  tables, model evaluation, or other analyses.
- has_icu_readmission_7days: after linkage, indicates an unplanned ICU readmission within seven
  days. A linked stay with elective_surgery=True is excluded from this outcome.
- has_elective_icu_readmission_7days: indicates that an observed seven-day return was linked to an
  elective-surgery ICU stay and excluded from the readmission outcome.
- readmission_stay_linked: indicates whether the recorded readmission time matched an exported
  follow-up stay. For study-cohort readmissions this should normally be True.
- readmission_elective_surgery: the linked follow-up stay's elective-surgery flag. It is nullable
  when no target stay can be linked.

The composite outcome is recalculated after linkage as death or unplanned ICU readmission within
seven days. Follow-up context rows are never themselves analyzed.
