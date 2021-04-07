from nwpipeline import nwpipeline as nwpl

study_dir = '/Volumes/KIT_DATA/test_study'

subject_id = '1027'
visit = '01'

nwpl.process_subject_visit(study_dir, subject_id, visit, quiet=False)
