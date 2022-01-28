from src import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_study'

test_nwpl = nwpl.NWPipeline(study_dir)
coll_ids = test_nwpl.get_coll_ids()

print(coll_ids)
