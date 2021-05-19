import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_ReMiNDD'

test_nwpl = nwpl.NWPipeline(study_dir)

#subject_ids = test_nwpl.get_subject_ids()
#coll_ids = test_nwpl.get_coll_ids()

#subject_ids = ['2707']
#coll_ids = ['01']

# print(subject_ids)
# print(coll_ids)

test_nwpl.run(single_stage=None, subject_ids=['8600'], overwrite_header=True, quiet=True, log=True)
