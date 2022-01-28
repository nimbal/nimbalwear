from src import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_ReMiNDD'

test_nwpl = nwpl.NWPipeline(study_dir)

#subject_ids = test_nwpl.get_subject_ids()
#coll_ids = test_nwpl.get_coll_ids()

subject_ids = ['1027']
#coll_ids = ['01']

# print(subject_ids)
# print(coll_ids)

test_nwpl.run(subject_ids=subject_ids, single_stage='gait', overwrite_header=True, quiet=True, log=True)