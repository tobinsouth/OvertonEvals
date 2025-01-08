# Evals for Measuring Overton Pluralism

This repo contains code for evaluating the Overton Pluralism project.

This project proposes a method for using social choice methods to formalize Overton pluralistic alignment, a concept introduced in this [work](https://arxiv.org/pdf/2402.05070) where a model provides comprehensive, high-coverage responses, representing a spectrum of reasonable responses. This contrasts with alignment to a single viewpoint or a limited set of perspectives.

1. **Dataset**

This is processed in `habermas_machine_data_processing.ipynb` it's uses Michiel's habermas machine data and extracts only the questions and opinions. It produces the `habermas_machine_questions.csv` file in `data/`. The multiple opinions are collapsed into a single array for each question. We will be indexing on question / question_id a lot.

3.** LLM Response Generation**

In `generating_llm_responses.ipynb` we use questions dataset and generate responses from various LLMs to be saved in `habermas_machine_questions_with_responses.csv`. This requires API keys for various LLMs which need to be added to the `.env` file.

4.** LLM Perspective Generation**

If we want to be able to hill climb, we also need the LLM to be able to generate a structured diversity of all perspectives, which we can evaluate against the human generated perspectives. This is done in `generating_llm_perspectives.ipynb` and saves `habermas_machine_questions_with_perspectives.csv`.

5. **Comparing Entailment Methods**

Here is the meat of the project. We want to determine in the LLM response contains an opinion, and find the entailment of the opinion in the response. Currently `evaluating_overton.ipynb` is performing this extraction and entailment, as well as performing some meta-analysis and testing. This is a very slow notebook to run (LLM calls) and needs to undergo a lot of work (and likely be extracting into some working files for distributed & offline processing).

6. Future: Applying the best entailment method across the full human and LLM generation opinion sets on all questions. Then, doing some analysis of the outcomes across models, and refining the metrics.

## Contributing

We welcome contributions to this project, which can be done via issues or pull requests, and also by emailing tsouth at mit edu.
