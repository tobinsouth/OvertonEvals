# Evals for Measuring Overton Pluralism

This repo contains code for evaluating the Overton Pluralism project.

This project proposes a method for using social choice methods to formalize Overton pluralistic alignment, a concept introduced in this [work](https://arxiv.org/pdf/2402.05070) where a model provides comprehensive, high-coverage responses, representing a spectrum of reasonable responses. This contrasts with alignment to a single viewpoint or a limited set of perspectives.

1. **Datasets with Questions and Human Perspectives**

This project draws data from a variety of sources to create a dataframe of [`questions`, `perspectives`, `source`]. The `perspectives` column is a list of strings, each representing a perspective, and is either directly human generated or a synthesised set of themes/perspectives from the dataset. Details vary by dataset (see below). 

All datasets are processed in `datagen_*.ipynb` and intermediate files are saved in `data/`. The final output is `questions_and_human_perspectives.csv` in `data/`.

2. **LLM Response Generation**

In `generating_llm_responses.ipynb` we use the above `questions_and_human_perspectives.csv` dataset and generate responses from various LLMs to be saved in `questions_and_human_perspectives_with_responses.csv`. This requires API keys for various LLMs which need to be added to the `.env` file.

4. **LLM Perspective Generation** [Incomplete]

If we want to be able to hill climb, we also need the LLM to be able to generate a structured diversity of all perspectives, which we can evaluate against the human generated perspectives. This is done in `generating_llm_perspectives.ipynb` and saves `questions_and_human_perspectives_with_perspectives.csv`.

5. **Comparing Entailment Methods**

Here is the meat of the project. We want to determine in the LLM response contains an opinion, and find the entailment of the opinion in the response. Currently `evaluating_overton.ipynb` is performing this extraction and entailment, as well as performing some meta-analysis and testing. This is a very slow notebook to run (LLM calls) and needs to undergo a lot of work (and likely be extracting into some working files for distributed & offline processing).

6. **Applying Entailment Methods**

[TODO:Elinor]

7. **Analysis**

[TODO:Tobin?]


## Data Source Notes

**Habermas Machine Data:** This is processed in `habermas_machine_data_processing.ipynb` it's uses Michiel's habermas machine data and extracts only the questions and opinions. It produces the `habermas_machine_questions.csv` file in `data/`.

**ValuePrism:** This is processed in `datagen_ValuePrism.ipynb` it's uses the [ValuePrism](https://huggingface.co/datasets/allenai/ValuePrism) dataset.


## Contributing

We welcome contributions to this project, which can be done via issues or pull requests, and also by emailing tsouth at mit edu.
