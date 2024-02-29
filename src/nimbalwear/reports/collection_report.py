from pathlib import Path
from datetime import timedelta, datetime

import pandas as pd
import toml
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib.patches import Patch as mpatch

#from .. import Study
from ..utils import read_excel_pwd

def collection_report(study_dir, subject_id, coll_id, supp_pwd=None, include_supp=True,
                      include_custom=True, daily_plot=True, fig_size=(18, 12), top_y=(0.25, 1), bottom_y=(0, 0.2),
                      supp_path=None):

    event_chars = pd.DataFrame([['sptw_night', 'Sleep period (night)', top_y, (0, 1), 0.5, 'mediumslateblue', None, None, 0, ],
                                ['sptw_day', 'Sleep period (day)', top_y, (0, 1), 0.1, 'mediumslateblue', None, None, 0, ],
                                ['sleep_night', 'Sleep (night)', top_y, (0.1, 0.9), 0.5, 'darkslateblue', None, None, 0, ],
                                ['sleep_day', 'Sleep (day)', top_y, (0.1, 0.9), 0.1, 'darkslateblue', None, None, 0, ],
                                ['chest_nonwear', 'Chest nonwear', top_y, (0.78, 0.9), 0.5, 'grey', None, None, 0, ],
                                ['chest_nonwear_crop', 'Chest nonwear (cropped)', top_y, (0.78, 0.9), 0.5, 'mistyrose', None, None, 0, ],
                                ['chest_no_collect', 'Chest not collecting', top_y, (0.78, 0.9), 0.5, 'brown', None, None, 0, ],
                                ['chest_sync', 'Chest sync point', top_y, (0.78, 0.9), 0.5, 'violet', None, None, 0],
                                ['lwrist_nonwear', 'Left wrist nonwear', top_y, (0.61, 0.73), 0.5, 'grey', '---', 'black', 0, ],
                                ['lwrist_nonwear_crop', 'Left wrist nonwear (cropped)', top_y, (0.61, 0.73), 0.5, 'mistyrose', '---', 'lightgrey', 0, ],
                                ['lwrist_no_collect', 'Left wrist not collecting', top_y, (0.61, 0.73), 0.5, 'brown', '---', 'lightgrey', 0, ],
                                ['lwrist_sync', 'Left wrist sync point', top_y, (0.61, 0.73), 0.5, 'violet', '---', 'black', 0],
                                ['rwrist_nonwear', 'Right wrist nonwear', top_y, (0.44, 0.56), 0.5, 'grey', '|||', 'black', 0, ],
                                ['rwrist_nonwear_crop', 'Right wrist nonwear (cropped)', top_y, (0.44, 0.56), 0.5, 'mistyrose', '|||', 'lightgrey', 0, ],
                                ['rwrist_no_collect', 'Right wrist not collecting', top_y, (0.44, 0.56), 0.5, 'brown', '|||', 'lightgrey', 0, ],
                                ['rwrist_sync', 'Right wrist sync point', top_y, (0.44, 0.56), 0.5, 'violet', '---', 'black', 0],
                                ['lankle_nonwear', 'Left ankle nonwear', top_y, (0.27, 0.39), 0.5, 'grey', '\\\\\\', 'black', 0, ],
                                ['lankle_nonwear_crop', 'Left ankle nonwear (cropped)', top_y, (0.27, 0.39), 0.5, 'mistyrose', '\\\\\\', 'lightgrey', 0, ],
                                ['lankle_no_collect', 'Left ankle not collecting', top_y, (0.27, 0.39), 0.5, 'brown', '\\\\\\', 'lightgrey', 0, ],
                                ['lankle_sync', 'Left ankle sync point', top_y, (0.27, 0.39), 0.5, 'violet', '\\\\\\', 'black', 0],
                                ['rankle_nonwear', 'Right ankle nonwear', top_y, (0.1, 0.22), 0.5, 'grey', '//////', 'black', 0, ],
                                ['rankle_nonwear_crop', 'Right ankle nonwear (cropped)', top_y, (0.1, 0.22), 0.5, 'mistyrose', '//////', 'lightgrey', 0, ],
                                ['rankle_no_collect', 'Right ankle not collecting', top_y, (0.1, 0.22), 0.5, 'brown', '//////', 'lightgrey', 0, ],
                                ['rankle_sync', 'Left ankle sync point', top_y, (0.1, 0.22), 0.5, 'violet', '//////', 'black', 0],
                                ['l_light_activity', 'Left light activity', top_y, (0.61, 0.73), 0.5, 'gold', None, None, 0, ],
                                ['l_moderate_activity', 'Left moderate activity', top_y, (0.61, 0.73), 0.5, 'orange', None, None, 0, ],
                                ['l_vigorous_activity', 'Left vigorous activity', top_y, (0.61, 0.73), 0.5, 'orangered', None, None, 0, ],
                                ['r_light_activity', 'Right light activity', top_y, (0.44, 0.56), 0.5, 'gold', None, None, 0, ],
                                ['r_moderate_activity', 'Right moderate activity', top_y, (0.44, 0.56), 0.5, 'orange', None, None, 0, ],
                                ['r_vigorous_activity', 'Right vigorous activity', top_y, (0.44, 0.56), 0.5, 'orangered', None, None, 0, ],
                                ['gait', 'Gait', top_y, (0.1, 0.39), 0.5, 'green', None, None, 0, ],],
                               columns=['event', 'label', 'level_y', 'rel_y', 'alpha', 'color', 'hatch', 'hatchcolor', 'default_duration', ])

    custom_chars = pd.DataFrame([['removal', 'Device removal (logged)', bottom_y, (0, 1), 0.5, 'grey', None, None, 0, ],
                                 ['in_bed', 'In bed (logged)', bottom_y, (0.1, 0.9), 0.5, 'mediumslateblue', None, None, 0, ],
                                 ['lights_out', 'Lights out (logged)', bottom_y, (0.3, 0.7), 0.5, 'darkslateblue', None, None, 0, ],
                                 ['nap', 'Nap (logged)', bottom_y, (0.3, 0.7), 0.1, 'darkslateblue', None, None, 0, ],
                                 ['activity', 'Activity (logged)', bottom_y, (0.3, 0.7), 0.5, 'gold', None, None, 0, ],
                                 ['meds', 'Medication', top_y, (0, 1), 0.5, 'deeppink', None, None, 300, ],],
                                columns = ['event', 'label', 'level_y', 'rel_y', 'alpha', 'color', 'hatch', 'hatchcolor', 'default_duration', ])

    if include_custom:
        event_chars = pd.concat([event_chars, custom_chars])

    event_chars = event_chars.set_index('event')

    #############################
    # Study info
    #############################

    study_dir = Path(study_dir)
    # study = Study(study_dir)
    # study_code = study.study_code
    # dirs = study.dirs

    study_code = study_dir.name

    settings_path = study_dir / 'study/settings/settings.toml'
    with open(settings_path, 'r') as f:
        settings_toml = toml.load(f)

    dirs = settings_toml['study']['dirs']
    dirs = {key: study_dir / value for key, value in dirs.items()}

    collections_csv_path = dirs['study'] / 'collections.csv'
    collection_info = pd.read_csv(collections_csv_path, dtype=str)

    devices_csv_path = dirs['study'] / 'devices.csv'
    device_info = pd.read_csv(devices_csv_path, dtype=str)

    print(f"\nSubject: {subject_id}...")

    ####################
    # Collection info
    ####################

    print('Formatting collection info...')
    collections = collection_info.copy()
    collections = collections[(collections['study_code'] == study_code) &
                              (collections['subject_id'] == subject_id) &
                              (collections['coll_id'] == coll_id)]
    collections.drop(columns=['study_code', 'subject_id', 'coll_id'], inplace=True)
    coll_info_html = collections.to_html(index=False)

    #####################
    # Supplementary info
    #####################
    # TODO: determine openpyxl package dependency?

    # Read clinical insights
    if include_supp:

        print("Formatting supplementary info...")
        if supp_path == None:
            supp_path = dirs['study'] / 'supplementary_info.xlsx'
        else:
            supp_path = Path(supp_path)

        if supp_pwd:
            supp = read_excel_pwd(supp_path, supp_pwd, dtype=str)
        else:
            supp = pd.read_excel(supp_path, dtype=str)

        supp.set_index(keys=['subject_id', 'coll_id'], inplace=True)

        if ((subject_id, coll_id) in supp.index):
            supp = supp.loc[(subject_id, coll_id)].to_frame()
            supp_html = supp.to_html(header=False, index_names=False)

    ######################
    # GET EVENT INFO
    #####################

    print("Formatting event data...")

    # initialize events dataframes
    detect_events = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time',
                                          'end_time', 'details', 'notes',])
    custom_events = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time',
                                          'end_time', 'details', 'notes',])
    events = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time', 'end_time',
                                   'details', 'notes', ])

    dtype_cols = {'study_code': str, 'subject_id': str, 'coll_id': str, 'event': str, 'id': pd.Int64Dtype(),}
    date_cols = ['start_time', 'end_time']

    # NONWEAR AND SYNC - PER DEVICE

    # get devices for this collection from device_list
    coll_device_list_df = device_info.loc[(device_info['study_code'] == study_code) &
                                          (device_info['subject_id'] == subject_id) &
                                          (device_info['coll_id'] == coll_id)]
    coll_device_list_df.reset_index(inplace=True, drop=True)

    # create reverse map of device location aliases
    loc_label_map = {}
    for dl_k, dl_v in settings_toml['pipeline']['device_locations'].items():
        for i in dl_v['aliases']:
            loc_label_map[i] = dl_k

    coll_device_list_df['dev_loc_label'] = [loc_label_map[r['device_location'].upper()]
                                            for i, r in coll_device_list_df.iterrows()]

    # initialize data
    coll_start = {}
    coll_end = {}
    nonwear = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time', 'end_time',
                                    'details', 'notes', ])
    sync = pd.DataFrame()

    # get nonwear and sync data for each device
    for i, r in coll_device_list_df.iterrows():

        # nonwear
        device_location = r['device_location']
        dev_loc_label = r['dev_loc_label']
        device_type = r['device_type']
        nonwear_csv_filename = f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}_NONWEAR.csv"
        nonwear_std_csv_path = dirs['nonwear_bouts_standard'] / nonwear_csv_filename
        nonwear_crp_csv_path = dirs['nonwear_bouts_cropped'] / nonwear_csv_filename
        nonwear_std = pd.read_csv(nonwear_std_csv_path, dtype=str)
        nonwear_crp = pd.read_csv(nonwear_crp_csv_path, dtype=str)

        nonwear_std['start_time'] = pd.to_datetime(nonwear_std['start_time'], yearfirst=True)
        nonwear_std['end_time'] = pd.to_datetime(nonwear_std['end_time'], yearfirst=True)
        coll_start[dev_loc_label] = nonwear_std['start_time'].iloc[0]
        coll_end[dev_loc_label] = nonwear_std['end_time'].iloc[-1]
        nonwear_std = nonwear_std[nonwear_std['event'] == 'nonwear']
        nonwear_label_map = {True: f'{dev_loc_label}_nonwear', False: f'{dev_loc_label}_nonwear_crop'}
        nonwear_std['event'] = (nonwear_std['id'].isin(nonwear_crp['id']).map(nonwear_label_map))
        nonwear_std['details'] = [f"{row['device_type']}_{row['device_location']}" for index, row in nonwear_std.iterrows()]
        nonwear_std['notes'] = ""
        #nonwear_std.drop(columns=['device_type', 'device_location'], inplace=True)
        nonwear_std = nonwear_std[['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time', 'end_time',
                                   'details', 'notes']]
        nonwear = pd.concat([nonwear, nonwear_std])
        nonwear.reset_index(inplace=True, drop=True)

        # sync
        sync_csv_filename = f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}_SYNC_EVENTS.csv"
        sync_csv_path = dirs['sync_events'] / sync_csv_filename
        if sync_csv_path.is_file():
            device_sync = pd.read_csv(sync_csv_path, dtype=str)
            device_sync['start_time'] = pd.to_datetime(device_sync['start_time'], yearfirst=True)
            device_sync['end_time'] = pd.to_datetime(device_sync['end_time'], yearfirst=True)
            sync = pd.concat([sync, device_sync])
            sync.reset_index(inplace=True, drop=True)

    sync_html = sync.to_html(index=False)

    # sync events
    if not sync.empty:
        sync_events = sync[sync['ref_sig_label'] != 'Config'].copy()

        sync_min_dur = pd.Timedelta('00:06:00')
        sync_events['duration'] = sync_events['end_time'] - sync_events['start_time']
        sync_events['pad'] = (sync_min_dur - sync_events['duration']) / 2
        sync_events['start_time'] = sync_events['start_time'] - sync_events['pad']
        sync_events['end_time'] = sync_events['end_time'] + sync_events['pad']
        #sync_events['event'] = [f"{r['device_location'].lower()}_sync" for i, r in sync_events.iterrows()]
        sync_events['event'] = [f"{loc_label_map[r['device_location'].upper()]}_sync" for i, r in sync_events.iterrows()]
        sync_events.rename(columns={'sync_id': 'id'}, inplace=True)
        sync_events['details'] = ""
        sync_events['notes'] = ""
        sync_events = sync_events[['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time', 'end_time',
                                        'details', 'notes', ]]

    else:
        sync_events = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time',
                                            'end_time',  'details', 'notes', ])

    # add no_collect event for when device was not collecting
    min_coll_start = min(coll_start.values())
    max_coll_end = max(coll_end.values())

    first_day_start = pd.Timestamp.combine(min_coll_start.date(), datetime.min.time())
    last_day_end = pd.Timestamp.combine(max_coll_end.date(), datetime.max.time())

    no_collect = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'event', 'id', 'start_time', 'end_time',
                                       'details', 'notes', ])

    for i, r in coll_device_list_df.iterrows():
        dev_loc_label = r['dev_loc_label']
        dev_no_collect = {'study_code': [study_code, study_code],
                          'subject_id': [subject_id, subject_id],
                          'coll_id': [coll_id, coll_id],
                          'id': [1, 2],
                          'event': [f"{dev_loc_label}_no_collect", f"{dev_loc_label}_no_collect"],
                          'start_time': [first_day_start, coll_end[dev_loc_label]],
                          'end_time': [coll_start[dev_loc_label], last_day_end],
                          'details': ["", ""],
                          'notes': ["", ""]}
        dev_no_collect = pd.DataFrame.from_dict(dev_no_collect)
        no_collect = pd.concat([no_collect, dev_no_collect])
        no_collect.reset_index(inplace=True, drop=True)

    # GAIT
    gait_csv_path = dirs['gait_bouts'] / f"{study_code}_{subject_id}_{coll_id}_GAIT_BOUTS.csv"
    gait_bouts = pd.read_csv(gait_csv_path, dtype=str)

    gait_bouts['event'] = 'gait'
    gait_bouts['start_time'] = pd.to_datetime(gait_bouts['start_time'], yearfirst=True)
    gait_bouts['end_time'] = pd.to_datetime(gait_bouts['end_time'], yearfirst=True)
    gait_bouts['details'] = gait_bouts['step_count'] + ' steps'
    gait_bouts.rename(columns={'gait_bout_num': 'id'}, inplace=True)
    gait_bouts.drop(columns=['step_count'], inplace=True)

    # SLEEP
    sptw_csv_path = dirs['sleep_sptw'] / f"{study_code}_{subject_id}_{coll_id}_SPTW.csv"
    sptw = pd.read_csv(sptw_csv_path, dtype=str)

    sleep_csv_path = dirs['sleep_bouts'] / f"{study_code}_{subject_id}_{coll_id}_SLEEP_BOUTS.csv"
    sleep_bouts = pd.read_csv(sleep_csv_path, dtype=str)

    sptw['start_time'] = pd.to_datetime(sptw['start_time'], yearfirst=True)
    sptw['end_time'] = pd.to_datetime(sptw['end_time'], yearfirst=True)
    sptw['relative_date'] = pd.to_datetime(sptw['relative_date'], format='%Y-%m-%d')
    sptw['overnight'] = sptw['overnight'].map({'True': True, 'False': False})
    sptw['event'] = ['sptw_night' if row['overnight'] else 'sptw_day' for index, row in sptw.iterrows()]

    sleep_bouts = sleep_bouts[sleep_bouts['bout_detect'] == 't8a4']
    sleep_bouts['start_time'] = pd.to_datetime(sleep_bouts['start_time'], yearfirst=True)
    sleep_bouts['end_time'] = pd.to_datetime(sleep_bouts['end_time'], yearfirst=True)
    sleep_bouts['overnight'] = sleep_bouts['overnight'].map({'True': True, 'False': False})
    sleep_bouts['event'] = ['sleep_night' if row['overnight'] else 'sleep_day' for index, row in sleep_bouts.iterrows()]

    sptw.rename(columns={'sptw_num': 'id'}, inplace=True)
    sptw.drop(columns=['relative_date', 'overnight'], inplace=True)

    sleep_bouts.rename(columns={'sleep_bout_num': 'id'}, inplace=True)
    sleep_bouts.drop(columns=['sptw_num', 'bout_detect', 'overnight'], inplace=True)

    # ACTIVITY
    activity_csv_path = dirs['activity_bouts'] / f"{study_code}_{subject_id}_{coll_id}_ACTIVITY_BOUTS.csv"
    activity_bouts = pd.read_csv(activity_csv_path, dtype=str)

    activity_bouts.rename(columns={'activity_bout_num': 'id', 'intensity': 'event'}, inplace=True)
    activity_bouts['start_time'] = pd.to_datetime(activity_bouts['start_time'], yearfirst=True)
    activity_bouts['end_time'] = pd.to_datetime(activity_bouts['end_time'], yearfirst=True)
    activity_bouts['details'] = [f"{r['device_location']}_{r['cutpoint_type']}" for i, r in activity_bouts.iterrows()]
    activity_bouts['side'] = [r['device_location'].lower()[0] for i, r in activity_bouts.iterrows()]
    activity_bouts = activity_bouts.loc[activity_bouts['event'].isin(['light', 'moderate', 'vigorous'])]
    activity_bouts['event'] = [f"{row['side']}_{row['event']}_activity" for idx, row in activity_bouts.iterrows()]
    activity_bouts.drop(columns=['device_location', 'cutpoint_dominant', 'cutpoint_type', 'side'], inplace=True)

    #TODO: split activity by side
    #TODO: add sedentary gait?

    # combine detected events
    detect_events = pd.concat([detect_events, nonwear, sync_events, no_collect, activity_bouts, sptw, sleep_bouts, gait_bouts])
    detect_events.sort_values(by='start_time', inplace=True)
    detect_events.reset_index(inplace=True, drop=True)

    # CUSTOM

    if include_custom:

        custom_events_csv_path = dirs['events_custom'] / f"{study_code}_{subject_id}_{coll_id}_EVENTS_CUSTOM.csv"

        if custom_events_csv_path.exists():
            custom_events = pd.read_csv(custom_events_csv_path, dtype=dtype_cols, parse_dates=date_cols)

    # combine all events
    events = pd.concat([events, detect_events, custom_events])
    events.sort_values(by='start_time', inplace=True)
    events.reset_index(inplace=True, drop=True)


    #####################################
    # daily event summary plot and table
    ######################################

    default_duration = pd.Series([timedelta(seconds=s) for s in event_chars.loc[events['event'], 'default_duration']])
    events['end_time'] = events['end_time'].fillna(events['start_time'] + default_duration)
    events['start_time'] = events['start_time'].fillna(events['end_time'] - default_duration)

    if daily_plot:

        print("Creating daily summary plot...")

        arb_date = datetime(year=1970, month=1, day=1).date()

        event_times = pd.concat([events['start_time'], events['end_time']])

        event_dates = sorted(list(set([x.date() for x in event_times if pd.isnull(x) == False])))

        start_date = min(event_dates)
        end_date = max(event_dates)

        numdays = (end_date - start_date).days + 1
        date_list = [start_date + timedelta(days=x) for x in range(numdays)]

        # create figure with subplots
        fig, axs = plt.subplots(nrows=numdays, ncols=1, sharex='all', figsize=fig_size)
        # fig.suptitle(f'Study: {study_code}, Subject: {subject_id}, Collection: {coll_id}\n{start_date} to {end_date}')

        # set minor and major tick intervals, date label format, and x limits
        min_hours = mdates.HourLocator(interval=1)
        maj_hours = mdates.HourLocator(interval=2)
        x_date_fmt = mdates.DateFormatter('%H:%M')

        axs[0].xaxis.set_minor_locator(min_hours)
        axs[0].xaxis.set_major_locator(maj_hours)
        axs[0].xaxis.set_major_formatter(x_date_fmt)
        axs[0].set_xlim([datetime.combine(arb_date, datetime.min.time()),
                         datetime.combine(arb_date, datetime.max.time())])

        plt.subplots_adjust(left=0.04, right=0.845, bottom=0.05, top=0.99, hspace=0.2, wspace=0.2)

        # remove all borders and ticks and display x grid
        for idx, date in enumerate(date_list):
            axs[idx].set_ylabel(date.strftime("%a\n%b %d"), labelpad=25, rotation=0, va='center')
            [s.set_visible(False) for s in axs[idx].spines.values()]
            [t.set_visible(False) for t in axs[idx].get_xticklines()]
            [t.set_visible(False) for t in axs[idx].get_yticklines()]
            [l.set_visible(False) for l in axs[idx].get_yticklabels()]
            axs[idx].grid(visible=True, which='both', axis='x', linestyle='--', color='grey')

        for idx, date in enumerate(date_list):

            day_start = datetime.combine(date, datetime.min.time())
            day_end = datetime.combine(date, datetime.max.time())
            day_events = events.loc[(events['end_time'] > day_start) & (events['start_time'] < day_end)].copy()

            day_events.loc[day_events['start_time'] < day_start, 'start_time'] = day_start
            day_events.loc[day_events['end_time'] > day_end, 'end_time'] = day_end

            day_events['start_time'] = [datetime.combine(arb_date, x.time()) for x in day_events['start_time']]
            day_events['end_time'] = [datetime.combine(arb_date, x.time()) for x in day_events['end_time']]

            for i, r in day_events.iterrows():

                if r['event'] in event_chars.index:
                    event_char = event_chars.loc[r['event']].copy()

                    level_y = event_char['level_y']
                    rel_y = event_char['rel_y']

                    level_range = level_y[1] - level_y[0]

                    event_min = level_y[0] + (rel_y[0] * level_range)
                    event_max = level_y[0] + (rel_y[1] * level_range)

                    end_time = r['end_time']
                    start_time = r['start_time']

                    axs[idx].axvspan(xmin=start_time, xmax=end_time, ymin=event_min, ymax=event_max,
                                     alpha=event_char['alpha'], fc=event_char['color'],
                                     hatch=event_char['hatch'], ec=event_char['hatchcolor'],
                                     linewidth=0)

        l = axs[0].legend()
        legend_handles = l.legendHandles
        for i, r in event_chars.iterrows():
            legend_handles.append(mpatch(fc=r['color'], label=r['label'], alpha=r['alpha'],
                                         hatch=r['hatch'], ec=r['hatchcolor']))
        # ncol = int(len(legend_handles) / 6) + 1
        axs[0].legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))

        figure_file_name = f'{study_code}_{subject_id}_{coll_id}_daily_events.png'
        figure_file_path = dirs['reports_collection'] / 'figures' / figure_file_name
        figure_file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_file_path)

        # create daily summary table

        print('Creating daily summary table...')
        file_name = lambda v: f"{study_code}_{subject_id}_{coll_id}_{v}.csv"

        # activity
        activity_path = dirs['activity_daily'] / file_name('ACTIVITY_DAILY')
        if activity_path.is_file():
            activity_csv = pd.read_csv(activity_path, dtype=str)
            activity_csv['date'] = pd.to_datetime(activity_csv['date'], yearfirst=True)
            activity_csv.drop(columns=['type', 'day_num', 'study_code', 'subject_id', 'coll_id', 'device_location',
                                       'cutpoint_type', 'cutpoint_dominant'], inplace=True)
            activity_csv.rename(columns={'none': 'no_activity'}, inplace=True)
        else:
            activity_csv = pd.DataFrame(columns=['date'])

        # gait
        gait_path = dirs['gait_daily'] / file_name('GAIT_DAILY')
        if gait_path.is_file():
            gait_csv = pd.read_csv(gait_path, dtype=str)
            gait_csv['date'] = pd.to_datetime(gait_csv['date'])
            gait_csv.drop(columns=['study_code', 'subject_id', 'coll_id', 'day_num', 'type'], inplace=True)
        else:
            gait_csv = pd.DataFrame(columns=['date'])

        # sleep
        sleep_path = dirs['sleep_daily'] / file_name('SLEEP_DAILY')
        if sleep_path.is_file():
            sleep_csv = pd.read_csv(sleep_path, dtype=str)
            sleep_csv['date'] = pd.to_datetime(sleep_csv['date'], yearfirst=True)
            sleep_csv = sleep_csv[(sleep_csv.bout_detect == 't8a4') & (sleep_csv.sptw_inc == 'all')]
            sleep_csv.drop(columns=['study_code', 'subject_id', 'coll_id', 'day_num', 'type', 'bout_detect',
                                    'sptw_inc'], inplace=True)
            sleep_csv.reset_index(inplace=True, drop=True)
        else:
            sleep_csv = pd.DataFrame(columns=['date'])

        summary_csv = pd.merge(pd.merge(activity_csv, gait_csv, on='date', how='outer'), sleep_csv, on='date',
                               how='outer')
        summary_html = summary_csv.to_html(index=False)

    #######################
    # Device information
    #######################

    print('Formatting device information...')

    calib_csvs = []
    for i, r in coll_device_list_df.iterrows():
        calib_name = f"{r['study_code']}_{r['subject_id']}_{r['coll_id']}_{r['device_type']}_{r['device_location']}_CALIB.csv"
        path = dirs['calib'] / calib_name
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            calib_csvs.append(df[['pre_err', 'post_err', 'iter']].values[0])
    calib_csvs = pd.DataFrame(calib_csvs, columns=['pre_err', 'post_err', 'iter'])
    devices_df = coll_device_list_df.drop(columns=['study_code', 'subject_id', 'coll_id']).reset_index(drop=True)

    devices_df = pd.concat([devices_df, calib_csvs], axis=1)
    devices_html = devices_df.to_html(index=False)

    ################
    # Compile report
    ###############

    print("Compiling report...")

    page_title_text = f'{study_code}_{subject_id}_{coll_id} collection report'
    title_text = f'{study_code}_{subject_id}_{coll_id}'
    date_range = f'{start_date} to {end_date}'
    rel_figure_path = Path('figures') / figure_file_name

    html = f'''
        <html>
            <head>
                <title>{page_title_text}</title>
                <style>
                    table {{border-collapse: collapse;}}
                    tbody {{border-top: 1px solid #1a1a1a !important; border-bottom: 1px solid #1a1a1a !important;}}
                    th, td {{padding: 5px;}}
                </style>
            </head>
            <body>
                <h1>{title_text}</h1>
                <h2>{date_range}</h2>
        '''

    html += f'''
            <h2>Collection information</h2>
            {coll_info_html}
            '''

    if include_supp:
        html += f'''
            <h2>Supplementary information</h2>
            {supp_html}
            '''

    html += f'''
        <h2>Daily events</h2>
        <img src={rel_figure_path} width='1600'>
        '''

    html += f'''
        <h2>Daily summary</h2>
        {summary_html}'''

    html += f'''
        <h2>Device information</h2>
        {devices_html}
        <h3>Sync events</h3>
        {sync_html}'''

    html += f'''
        </body>
        </html>
        '''

    html_report_name = f'{study_code}_{subject_id}_{coll_id}_collection_report.html'

    html_report_path = dirs['reports_collection'] / html_report_name
    html_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_report_path, 'w') as f:
        f.write(html)
    print(f"Report saved: {html_report_path}")

    #    # ]\[[[[[[[=o[}/;] - Emils, 3 Dec 2021
    #    # }{}][]
    #


