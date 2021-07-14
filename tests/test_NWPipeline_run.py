import nwpipeline as nwpl

study_dir = r'W:\NiMBaLWEAR\test-HANDDS'

test_nwpl = nwpl.NWPipeline(study_dir)

#subject_ids = test_nwpl.get_subject_ids()
#coll_ids = test_nwpl.get_coll_ids()

subject_ids = ['061621']
#coll_ids = ['01']

# print(subject_ids)
# print(coll_ids)

test_nwpl.run(subject_ids=subject_ids, overwrite_header=True, gait_axis=0, quiet=True, log=True)
