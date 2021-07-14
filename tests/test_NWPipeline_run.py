import nwpipeline as nwpl

study_dir = r'W:\NiMBaLWEAR\OND08'

test_nwpl = nwpl.NWPipeline(study_dir)

#subject_ids = test_nwpl.get_subject_ids()
#coll_ids = test_nwpl.get_coll_ids()

subject_ids = ['0004', '0007', '0009', '0012']
#coll_ids = ['01']

# print(subject_ids)
# print(coll_ids)

test_nwpl.run(subject_ids=subject_ids, single_stage='sleep', overwrite_header=True, gait_axis=0, quiet=True, log=True)
