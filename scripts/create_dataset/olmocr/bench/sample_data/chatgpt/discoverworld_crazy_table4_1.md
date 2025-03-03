Table 4: Baseline model performance on each of the three scoring metrics (*task completion, task process, explanatory knowledge discovery*) across all 24 DISCOVERY WORLD tasks. Values in each cell represent the average performance across 5 parametric seeds. Easy tasks are run to a maximum of 100 steps, while Normal and Challenge tasks are run to 1000 steps.

| #  | Topic                        | Task                              | ReACT | Plan+Execute | Hypothesizer |
|----|------------------------------|-----------------------------------|-------|--------------|--------------|
| 1  | Proteomics                   | Clustering                        |       |              |              |
| 2  | Easy                         | Simplified Clustering             | 0.87  | 0.20         | 0.08         |
| 3  | Normal                       | Clustering (2D)                   | 0.88  | 0.40         | 0.68         |
| 4  | Challenge                    | Clustering (3D)                   | 0.88  | 0.40         | 0.60         |
| 5  | Chemistry                    | Exploring Combinations and Hill Climbing |       |              |              |
| 6  | Easy                         | Single substances                 | 0.87  | 1.00         | 0.70         |
| 7  | Normal                       | Mix of 3 substances               | 0.82  | 0.00         | 0.87         |
| 8  | Challenge                    | Mix of 4 substances               | 0.40  | 0.00         | 0.87         |
| 9  | Archaeology                  | Correlations                      |       |              |              |
| 10 | Easy                         | Simple instrument                 | 0.87  | 0.60         | 0.33         |
| 11 | Normal                       | Instrument Use                    | 0.72  | 0.40         | 0.77         |
| 12 | Challenge                    | Correlation                       | 0.40  | 0.00         | 0.05         |
| 13 | Reactor Lab                  | Regression                        |       |              |              |
| 14 | Easy                         | Linear regression                 | 0.42  | 0.00         | 0.10         |
| 15 | Normal                       | Quadratic regression              | 0.43  | 0.00         | 0.00         |
| 16 | Challenge                    | Uncovering systems of rules       |       |              |              |
| 17 | Easy                         | Simplified rules                  | 0.91  | 0.60         | 0.78         |
| 18 | Normal                       | Presence rules                    | 0.87  | 0.00         | 0.40         |
| 19 | Challenge                    | Legal Rules                       | 0.00  | 0.00         | 0.00         |
| 20 | Translation                  | Rosetta-stone style linguistic discovery of alien language |       |              |              |
| 21 | Easy                         | Noun and verb                     | 0.57  | 0.00         | 0.00         |
| 22 | Normal                       | Noun, adj., and verb              | 0.49  | 0.00         | 0.55         |
| 23 | Challenge                    |                                   |       |              |              |
| 24 | Average (Easy)               |                                   | 0.59  | 0.25         | 0.56         |
| 25 | Average (Normal)             |                                   | 0.05  | 0.14         | 0.18         |
| 26 | Average (Challenge)          |                                   | 0.63  | 0.10         | 0.15         |

Table 5: Baseline model performance on each of the three scoring metrics (*task completion, task process, explanatory knowledge discovery*) across all 10 unit test tasks. Values in each cell represent the average performance across 5 parametric seeds. Unit tests tasks are run to a maximum of 100 steps.

| #  | Unit Test Topic              | ReACT | Plan+Execute | Hypothesizer |
|----|------------------------------|-------|--------------|--------------|
| 25 | Multi-turn dialog with an agent | 1.00  | 1.00         | 1.00         |
| 26 | Measure an object with an instrument | 0.87  | 0.60         | 0.73         |
| 27 | Pick-and-place object        | 0.90  | 0.80         | 0.80         |
| 28 | Pick-and-place object        | 1.00  | 1.00         | 1.00         |
| 29 | Read DiscoveryFeed posts     | 1.00  | 1.00         | 1.00         |
| 30 | Move through doors           | 0.55  | 0.20         | 0.25         |
| 31 | Using keys with doors        | 0.60  | 0.20         | 0.00         |
| 32 | Navigate to a specific room in a house | 0.20  | 0.00         | 0.00         |
| 33 | Search an environment for an object | 0.80  | 0.00         | 0.00         |
| 34 | Interact with a moving agent | 0.80  | 0.20         | 0.53         |
| 35 | Average (Unit Tests)         | 0.76  | 0.60         | 0.66         |

4.2 Baseline Agent Models

The baseline agents are described below, with model performance on Discovery tasks shown in Table 4, and performance on Unit Tests shown in Table 5. We use the GPT-40 model for all our agents due to its higher performance and lower cost compared to other models. For space we provide...