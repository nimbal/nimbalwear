import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_feedback'

test_nwpl = nwpl.NWPipeline(study_dir)

test_nwpl.run(overwrite_header=True, quiet=True, log=True)