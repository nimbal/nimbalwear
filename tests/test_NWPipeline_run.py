import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test-HANDDS'
collections = [('060821', '01')]

test_nwpl = nwpl.NWPipeline(study_dir)

test_nwpl.run(collections=collections, single_stage=None, overwrite_header=True, gait_axis=0, quiet=True, log=True)
