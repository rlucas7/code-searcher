# steps in annotation workflow for humans

Step 0. Confirm you can run the app locally

Step 1. Load an indexed repository into the app via click cmd 'init-db' from the README file.

Choose one of the `*.jsonl` files from the project which you have not annotated yet,
an indexed repository is one of the `*.jsonl` files. For example `Collections-go.jsonl` contains the index of the go language repository.
NOTE: Eventually you will annotate all the 5 `*.jsonl` files.

Step 2. Register a user and login to the app to enable the annotations workflow

Each query beneath will-eventually-be executed against each indexed repository.

For every query in [the Queries](https://docs.google.com/spreadsheets/d/1ti_IqZZUtO8TS-RHvTSHgaUHC7CAoaM3Vl67xJhC8YY/edit?gid=0#gid=0):
	Step 3. Execute the query in the app

	Step 4. Mark each search result as relevant or not and click 'done' to submit to the relevance determination to the app.
	The clicking of done hides the html element from further relevance determination selections.
	The submission of the relevance determine *does not* require the done button to be clicked.

	NOTE: The relevance is automatically submitted on each selection of the yes/no and the most recent selection is retained in subsequent steps as the default-this is the `dupstrat` option which defaults to `takelast` in the `gen-llm-rels` command in step 7. Currently `takelast` is the *only* supported duplicate handling strategy. 

Once all the queries are executed and all relevance annotations are stored:

Step 5. Close the app and execute the command: `export-rad-to-csv` from the readme.
	This will export the csv file of the relevance annotations from the human.
	The naming convention on these files should be:
	`rad-<programming-lang>-<human-id>.csv` for example,
	`rad-c-lang-rlucas7.csv` would be the human relevances for the rlucas7 human.

Step 6. Upload the file from step 5 to the bucket by using the 'upload' button in the browser ui.

Step 7. For each AI (openai, gemini are the 2 currently supported), generate the relevance annotation determinations for the AI using the command `gen-llm-rels` from the README. 
The convention used for these output files is
`llm_gen_rel-<AI-name>-<programming-language>-<human-id>.csv` for example:
`llm_gen_rel-gemini-java-rlucas7.csv` is one set of generated AI & Human annnotations.

Step 8. Backup/upload this generated output file to the project output bucket.

Step 9. Generate the IR metrics for the output file frin steo 7, and copy the outputs from stdout and paste those output metrics in [the doc](https://docs.google.com/document/d/1V0IDoBw66FctPoFOjNgzAZ7i5j_s4hEoOTd-SOJCXpI/edit?tab=t.0). 

NOTE: The doc in Step 9 is a scratch doc and the ultimate results will be collated into a table in the manuscript.