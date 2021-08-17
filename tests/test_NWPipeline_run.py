import nwpipeline as nwpl

study_dir = r'W:\NiMBaLWEAR\test-HANDDS'
subject_ids = ['061621']

test_nwpl = nwpl.NWPipeline(study_dir)

test_nwpl.run(single_stage='gait', overwrite_header=True, gait_axis=0, quiet=True, log=True)
