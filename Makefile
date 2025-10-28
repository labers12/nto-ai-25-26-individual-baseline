.PHONY: install train predict validate run clean

# Use the current python interpreter
PYTHON := python

install:
	@echo "Assuming dependencies are managed by poetry. Use 'poetry install'."

train:
	@echo "Running training script..."
	$(PYTHON) -m src.nto_baseline.train

predict:
	@echo "Running prediction script..."
	$(PYTHON) -m src.nto_baseline.predict

validate:
	@echo "Running validation script..."
	$(PYTHON) -m src.nto_baseline.validate

run: train predict validate
	@echo "Full pipeline executed successfully."

clean:
	@echo "Cleaning output directories..."
	rm -f output/models/*
	rm -f output/submissions/*
	@echo "Done."
