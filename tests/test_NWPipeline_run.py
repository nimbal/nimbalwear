import nwpipeline as nwpl

study_dir = 'w:/NiMBaLWEAR/dev-OND09'
collections = [('0001','01'), ('0007','01'), ('0008','01'), ('0005','01'), ('0006','01')]
single_stage = None

test_nwpl = nwpl.Pipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=single_stage, quiet=True, log=True)
