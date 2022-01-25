from nwpipeline import Pipeline

study_dir = 'w:/dev/dev-nimbalwear/dev-OND09'
collections = [('0004','01'), ('0021','01'), ('0016','01'), ('0017','01'), ('0022','01'), ('0029','01'), ('0015','01'),
               ('0026','01'), ('0031','01'), ('0033','01'), ('0038','01')]
collections = [('0037', '01')]
single_stage = 'convert'

test_nwpl = Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
