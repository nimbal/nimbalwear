from nwpipeline import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_study'

subject_id = '1027'
visit = '01'

test_nwpl = nwpl.NWPipeline(study_dir)
test_nwpl.process_subject_visit(subject_id, visit, quiet=False)
