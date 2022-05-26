from src.nimbalwear import Pipeline

study_dir = 'w:/NiMBaLWEAR/OND09'
settings_path = 'W:/NiMBaLWEAR/OND09/pipeline/settings/settings.json'

collections = [('0189', '01')]
#collections = None
single_stage = 'convert'

ond09 = Pipeline(study_dir, settings_path)

ond09.run(collections, single_stage=single_stage, quiet=True, log=True)
