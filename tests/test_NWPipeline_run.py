import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_PD_DANCE'

subject_ids = ['0001']
coll_ids = ['01']

test_nwpl = nwpl.NWPipeline(study_dir)
test_nwpl.run(subject_ids, coll_ids, overwrite_header=True, quiet=True)