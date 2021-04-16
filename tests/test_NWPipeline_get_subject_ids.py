import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_study'

subject_ids = ['1027']
coll_ids = ['01']

test_nwpl = nwpl.NWPipeline(study_dir)
subject_ids = test_nwpl.get_subject_ids()

print (subject_ids)
#test_nwpl.run()