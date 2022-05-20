from src.nimbalwear import Pipeline

study_dir = 'w:/NiMBaLWEAR/OND09'

collections = [('0149', '01'), ('0150', '01')]
#collections = None
single_stage = None

ond09 = Pipeline(study_dir)


ond09.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
