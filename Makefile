PY=python
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
data:
	$(PY) scripts/00_make_sft_dataset_zarathustra.py
train:
	BOOKSFT_BASE=meta-llama/Meta-Llama-3.1-8B-Instruct $(PY) scripts/01_train_sft.py --config configs/sft.yaml
merge:
	$(PY) scripts/02_merge_adapter.py
eval:
	$(PY) scripts/03_eval_perplexity.py
	$(PY) scripts/04_eval_sft_qa.py
app:
	streamlit run app/streamlit_app.py
