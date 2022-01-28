from src import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_study'

test_nwpl = nwpl.NWPipeline(study_dir)
subject_ids = test_nwpl.get_subject_ids()

print(subject_ids)