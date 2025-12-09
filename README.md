### Project Details

Word Sense Disambiguation tasks commonly assume one word sense to be the 'correct' one, but that is not necessarily reflective of reality. Ambiguities, underspecification and personal opinions can influence which word senses one finds plausible in a given context, and there is a difference between the intuition of humans and the predictions of computational models. To study and benchmark this phenomenon, we introduce the AmbiStory dataset, a dataset of 5-sentence short stories. The task is to disambiguate a target homonym in the fourth sentence through contextual clues in surrounding sentences.

Our stories consist of three parts: A precontext, consisting of three sentences that ground the story, an ambiguous sentence, containing a homonym that causes it to have two widely different plausible interpretations, and optionally one of two endings, which often imply a specific word sense of the homonym.

We asked Prolific participants to rate the plausibility of a given word sense in the context of a story on a scale from 1 to 5. We collect at least five ratings for each sense/story sample. Since each story setup has either one of two endings or none at all, and we focus on two word senses per homonym, we obtain six annotation samples per setup.

- Examples are in the Data category of this readme

In essence, the task is to predict the human-perceived plausibility of a word sense by selecting a score between 1 and 5. We evaluate this using two primary metrics:

Spearman Correlation: The predicted plausibility score should ideally correlate to the average of scores assigned by humans.
Accuracy Within Standard Deviation: Some samples have a clearer consensus than others, so we calculate accuracy as the proportion of model predictions that are within standard deviation (at least 1) from the average judgment by annotators. This is considered the main metric on the leaderboard, but please try to optimize both equally.

### Initial Links

- [Project Details Page](https://nlu-lab.github.io/semeval.html)
- [More Detailed Project Page](https://www.codabench.org/competitions/10877/#/pages-tab)
- [Sample Data](https://drive.google.com/drive/folders/1P1mPpl6-wsNkXULgkFXDGFrx92Mqy-M3)
- [Training & Validation Data](https://drive.google.com/drive/folders/1evACvNGyBPKr99R5Db-zo2sK4u2_a9i3)
- [Baseline & Evaluation](https://github.com/Janosch-Gehring/semeval26-05-scripts)

#

### Phases

- Initial Progress Presentation (5 points)
- Project Work (30 points)
- Project Presentation (5 points)

#

### To-Do

- Understand the task (1/2)
- Look up the data and understand it (1/2)
- Understand the evaluation metrics (1/2)
- Understand the scoring tool (1/2)
- Plan on what you might do
- Complete the form for initial progress presentation
- Present the initial progress

# TBD

# Data

The dataset consists of narrative samples, each representing a short 5-sentence story containing an ambiguous homonym. 
The training set contains 2,280 samples, each described by 12 variables.

![Image1](aed/var.jpg)


- Overall data format 
```json
{
    "1":{...},
    "2":{...},
    "3":{...},
    "4":{...},
    "5":{...},
    "6":{...}
}
```

- Entity format
```json
"1": {
        "homonym": "bugs", # This is the word we have to find the meaning of
        "judged_meaning": "general term for any insect or similar creeping or crawling invertebrate", # the meaning on which the 
        # grades were given for the context 
        "precontext": "Anna was having a tough week. Her room was a mess, and her computer kept crashing. Frustrated by everything going wrong, she called Jen.", # 3 sentences with the base story
        "sentence": "She asked her friend to help her get rid of the bugs.", # the sentence that might be misleading
        "ending": "They were crawling on the keyboard. Maybe that was the reason it didn't work.", # the ending
        "choices": [
            5,
            5,
            1,
            2,
            5
        ], # 5 people / 5 ratings
        "average": 3.6, # the avg
        "stdev": 1.949358868961793, 
        "nonsensical": [
            false,
            false,
            false,
            false,
            false
        ],
        "example_sentence": "The garden was full of bugs."
    },
    "2": {
        "homonym": "bugs",
        "judged_meaning": "a fault or defect in a computer program, system, or machine",
        "precontext": "Anna was having a tough week. Her room was a mess, and her computer kept crashing. Frustrated by everything going wrong, she called Jen.",
        "sentence": "She asked her friend to help her get rid of the bugs.",
        "ending": "They were crawling on the keyboard. Maybe that was the reason it didn't work.",
        "choices": [
            1,
            1,
            2,
            1,
            5
        ],
        "average": 2.0,
        "stdev": 1.7320508075688772,
        "nonsensical": [
            false,
            false,
            false,
            false,
            false
        ],
        "example_sentence": "There's a bug in the software."
    },
    "3": {
        "homonym": "bugs",
        "judged_meaning": "general term for any insect or similar creeping or crawling invertebrate",
        "precontext": "Anna was having a tough week. Her room was a mess, and her computer kept crashing. Frustrated by everything going wrong, she called Jen.",
        "sentence": "She asked her friend to help her get rid of the bugs.",
        "ending": "Installing new antivirus software didn't do the trick.",
        "choices": [
            1,
            1,
            1,
            1,
            1
        ],
        "average": 1.0,
        "stdev": 0.0,
        "nonsensical": [
            false,
            false,
            false,
            false,
            false
        ],
        "example_sentence": "The garden was full of bugs."
    },
    "4": {
        "homonym": "bugs",
        "judged_meaning": "a fault or defect in a computer program, system, or machine",
        "precontext": "Anna was having a tough week. Her room was a mess, and her computer kept crashing. Frustrated by everything going wrong, she called Jen.",
        "sentence": "She asked her friend to help her get rid of the bugs.",
        "ending": "Installing new antivirus software didn't do the trick.",
        "choices": [
            5,
            5,
            5,
            5,
            4
        ],
        "average": 4.8,
        "stdev": 0.4472135954999579,
        "nonsensical": [
            false,
            false,
            false,
            false,
            false
        ],
        "example_sentence": "There's a bug in the software."
    },
    "5": {
        "homonym": "bugs",
        "judged_meaning": "general term for any insect or similar creeping or crawling invertebrate",
        "precontext": "Anna was having a tough week. Her room was a mess, and her computer kept crashing. Frustrated by everything going wrong, she called Jen.",
        "sentence": "She asked her friend to help her get rid of the bugs.",
        "ending": "",
        "choices": [
            2,
            2,
            2,
            5,
            1
        ],
        "average": 2.4,
        "stdev": 1.51657508881031,
        "nonsensical": [
            false,
            false,
            false,
            false,
            false
        ],
        "example_sentence": "The garden was full of bugs."
    },
    "6": {
        "homonym": "bugs",
        "judged_meaning": "a fault or defect in a computer program, system, or machine",
        "precontext": "Anna was having a tough week. Her room was a mess, and her computer kept crashing. Frustrated by everything going wrong, she called Jen.",
        "sentence": "She asked her friend to help her get rid of the bugs.",
        "ending": "",
        "choices": [
            4,
            5,
            4,
            5,
            5
        ],
        "average": 4.6,
        "stdev": 0.5477225575051661,
        "nonsensical": [
            false,
            false,
            false,
            false,
            false
        ],
        "example_sentence": "There's a bug in the software."
    },

```
- Missing values
<p align="center"> <img src="aed/missingno_matrix_001.png" width="70%" /> </p>

### Exploratory Data Analysis (EDA)
#### Numerical variables analysis
- We visualized distributions of average, standard deviation, and individual choices, 
which show that  choices are slightly biased toward higher ratings, and standard deviation remains low for most samples, indicating strong annotator agreement.

<p float="left">
  <img src="aed/hist_average_002.png" width="48%" />
  <img src="aed/hist_stdev_003.png" width="48%" />
</p>
<p align="center">
  <img src="aed/hist_choices_004.png" width="48%">
</p>

- The scatterplots show that standard deviation varies more widely, indicating differing levels of annotator agreement across samples.

<p float="left">
  <img src="aed/scatter_avg_choices_009.png" width="48%" />
  <img src="aed/scatter_stdev_choices_010.png" width="48%" />
</p>

- A correlation heatmap was computed for the numerical variables (average, stdev, choices), showing a 
strong positive correlation between choices and average.

<p align="center">
  <img src="aed/heatmap_corr_011.png" width="38%">
</p>

#### Textual variables analysis
- We visualized the distribution of homonym,the most frequent homonyms appearing between 25 and 60 times in the dataset.

<p align="center"> 
    <img src="aed/homonym_counts_012.png" width="70%" /> 
</p>

- We examined how many distinct meanings each homonym has. Most frequent homonyms exhibit between 3 and 5 senses,
the dataset covering a wide range of semantic ambiguity.

<p align="center"> 
    <img src="aed/meanings_per_homonym_013.png" width="70%" /> 
</p>

- We analyzed how plausibility scores varies across different senses of the same homonym. The boxplots show substantial variability within many homonyms, indicating that some senses are consistently rated as more plausible than others depending on the narrative context.

<p align="center"> <img src="aed/homonyms_average_017.png" width="70%" /> </p>

- We explored whether the presence of a story ending influences plausibility scores. 
The distribution of average ratings is very similar, suggesting that endings do not strongly affect how plausible a sense is perceived in most cases.

<p align="center"> <img src="aed/has_ending_018.png" width="48%" /> </p>

#### Nonsensical annotations analysis

We analyzed how annotators’ nonsensical judgments relate to the plausibility scores and the level of agreement across samples.

- Only a small fraction received 1–3 nonsensical flags, indicating that implausible senses are relatively rare.

<p align="center"> <img src="aed/count_nonsense_014.png" width="48%" /> </p>

- As the number of judgments increases, the average plausibility decreases and the standard deviation becomes higher, 
indicating that annotators assign lower scores and disagree more when a sense is perceived as implausible.
<p float="left">
  <img src="aed/nonsense_average_015.png" width="48%" />
  <img src="aed/nonsense_stdev_016.png" width="48%" />
</p>

### The evaluation

We use an evaluation project given in the project's description

Here lies a folder `input/ref` and `input/res`

- `input/ref/solutions.jsonl` contains the actual factual results

- `input/res/predictions.jsonl` contains the predictions made by the project

The solutions look like

```json
{"id": "0", "label": [4, 5, 3, 1, 5]}
{"id": "1", "label": [3, 3, 4, 4, 4]}
{"id": "2", "label": [5, 5, 2, 3, 4]}
```

The predictions must look like

```json
{"id": "0", "prediction": 2}
{"id": "1", "prediction": 2}
{"id": "2", "prediction": 1}
```


To test everything works fine you have to:

- Ideally create an env
- pip install -r requirements.txt
- Create an output folder
- change the predictions (keep the solution as is because we run on the dev dataset)
- run python scoring.py input/ref/solution.jsonl input/res/predictions.jsonl output/scores.json
= Interpret the results

```bash
(venv) PS D:\Master\Sem1\RN\semeval26-05-scripts> python scoring.py input/ref/solution.jsonl input/res/predictions.jsonl output/scores.json
Importing...
Starting Scoring script...
Everything looks OK. Evaluating file input/res/predictions.jsonl on input/ref/solution.jsonl
----------
Spearman Correlation: 0.04940429751709947
Spearman p-Value: 0.2316295121977508
----------
Accuracy: 0.4608843537414966 (271/588)
Results dumped into scores.json successfully.
```

From the solutions given by the human we can extract:

- the mean (average): sum([1,2,3,4,5]) / 5
- stddev: how different were the values given by each human
stdev([1,1,1,1,1]) = 0
stdev([1,1,1,1,10]) = 3.3...

What does these 2 evalutation metrics mean? (spearman correlation, accuracy within SD)?

- Spearman Correlation: How similarly two lists are ordered. If humans rank examples (avg) Id1 > ID2 > ID3 and the model does as well the value is close to 1. If the order is reversed then -1. If the order is not consistent then the value is 0.

- We also have Spearman p-value: It means how likely is it that this correlation happened by random chance. The smaller the value the more real the correlation is, if the value is high then the correlation might be random

- Accuracy Within Standard Deviation: How close are the predictions to the humans, where "close" is highly related to how much the annotators dissagred. It has 3 cases possible

1. If prediction is strictly inside the interval (avg − stdev) < prediction < (avg + stdev) → return True
2. Else if absolute distance |avg − prediction| < 1 → return True
3. Else → return False

Lets picture an example

### Example 1

Labels: [3,3,3,3,3]
avg = 3.0
stdev = 0.0
prediction 3.0 

- First check false (3.0 is not strictly between 3.0 and 3.0)
- Second check |3−3|=0<1 → True

prediction 4.0

- First check failes
- Second check |3−4|= 1.0 → second check false (strict <1) → False (wrong)

### Example 2

Labels: [1,2,3,4,5]
avg = 3.0
stdev ≈ 1.581
prediction 4.4

- Falls inside (3−1.581, 3+1.581) = (1.419, 4.581) so first check True
- We dont get to the second check, which would be false |3−4.4|=1.4 (>1)


## Baseline

For now we have 2 baselines:

- Majority - as the name implies contains only the prediction 4 as it was the value mostly given by the people that took part in the projects dataset creation 

- Random - random predictions