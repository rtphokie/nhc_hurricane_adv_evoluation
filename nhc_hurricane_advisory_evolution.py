'''
https://www.nhc.noaa.gov/gis/
'''
author = "@rtphokie"
from io import StringIO
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from os import listdir
from os.path import isfile, join
from pprint import pprint
from tqdm import tqdm
import datetime
import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os,io
import pathlib
import pytz
import requests
import shapefile
import unittest
import warnings
import zipfile
cdict = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0))}

cmap_g2r = mcolors.LinearSegmentedColormap( 'my_colormap', cdict, 200)

def get_year(year=2019):
    jkl = None

def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test

def make_map(projection='mill', resolution='l', background=None,
             llcrnrlon=-85., llcrnrlat=10.,
             urcrnrlon=-45., urcrnrlat=45.,
             ):
    m = Basemap(projection=projection, lon_0=-78., resolution=resolution,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                )
    if background == 'BlueMarble':
        # m.bluemarble()
        m.shadedrelief()
    else:
        m.fillcontinents(color='#cc9966', lake_color='#99ffff')
        m.drawmapboundary(fill_color='#99ffff')
    m.drawparallels(np.arange(10, 70, 10), alpha=0.4, color='grey', labels=[1, 1, 0, 0])
    m.drawmeridians(np.arange(-100, 0, 10), alpha=0.4, color='grey', labels=[0, 0, 0, 1])
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    return m

def setup_directories(basedir):
    for dir in ['animations', 'frames', 'nhc_data/5day', 'nhc_data/fcst']:
        pathlib.Path(f"{basedir}/{dir}").mkdir(parents=True, exist_ok=True)

def get_advisories(dir, year, storm, basin='AL'):
    if year < 2008:
        raise Exception('NHC data begins in 2018 ')
    setup_directories(dir)
    #https://www.nhc.noaa.gov/gis/forecast/archive/al052019_5day_036.zip
    #https://www.nhc.noaa.gov/gis/forecast/archive/al052019_fcst_001.zip
    goget = []
    advisory = '0'
    # for prod in [ 'fcst']:
    for prod in ['5day', 'fcst']:
        pref = f"{basin}{storm:02d}{year}_{prod}".lower()
        # print(pref)
        for filename in sorted(os.listdir(f"{dir}/nhc_data/{prod}")):
            if pref in filename:
                advisory = filename.replace(f"{pref}_", '')
        if 'A' not in advisory:
            goget.append("%03dA" % int(advisory))
        goget.append("%03d" % (int(advisory.replace('A', '')) + 1))
        statuses=[]
        for s in goget:
            path = f"{dir}/nhc_data/{prod}/{pref}_{s}"
            zip_file_url = f"https://www.nhc.noaa.gov/gis/forecast/archive/{pref}_{s}.zip"
            # print(f"{path} {zip_file_url}")
            r = requests.get(zip_file_url, stream=True)
            statuses.append(r.status_code)
            if r.status_code < 300:
                cwd = os.getcwd()
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                print (f"created {path}")
                os.chdir(path)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall()
                os.chdir(cwd)

    if not list(set(statuses))[0] >= 300:
        # keep going until all downloads fail
        get_advisories(dir, year, storm, basin=basin)

def make_frames(dir, year, storm, basin='AL', fetch_latest_adv=True, overwrite=False):
    adv=f"{basin}{storm:02}{year}_5day"
    if not os.path.exists(dir):
        os.mkdir(dir)
    if fetch_latest_adv:
        get_advisories(dir,year, storm, basin=basin)

    data = get_data_from_shapefiles(dir)

    for advisnum, v in tqdm(sorted(data.items())):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pltfilename = f"{dir}/frames/{dir}_{advisnum}.png"
        if not overwrite and os.path.exists(pltfilename):
            continue
        m = make_map(projection='mill', background='BlueMarble')
        cnt=0
        pastadvs = previous_advisories(advisnum, data)
        # initialize off map with maximum to ensure early plots reflect a 0-200 scale
        lats = [0]
        lons = [0]
        winds = [200]
        prevcstr = ''
        catstr = ''
        for advisnum2 in tqdm(pastadvs):
            cnt +=1
            m.readshapefile(data[advisnum2]['lin']['filename'].replace('.shp', ''), 'past_track',
                                             linewidth=0.5, color= 'grey', zorder=1)
            m.readshapefile(data[advisnum2]['pts']['filename'].replace('.shp', ''), 'past_pts',
                                             linewidth=0.0)
            pos = data[advisnum2]['lin']['points'][0]
            lons.append(pos[0])
            lats.append(pos[1])
            wind = m.past_pts_info[0]['MAXWIND']
            if wind >= 157:
                catstr = 'cat 5'
            if wind >= 130:
                catstr = 'cat 4'
            if wind >= 111:
                catstr = 'cat 3'
            if wind >= 96:
                catstr = 'cat 2'
            if wind >= 74:
                catstr = 'cat 1'
            if wind >= 39:
                catstr = 'TS'
            if wind > 0:
                catstr = 'TD'
            if catstr != prevcstr:
                prevcstr = catstr
                print(catstr)
            winds.append(m.past_pts_info[0]['GUST'])
        x, y = m(lons, lats)
        # plt.scatter(x, y, c=winds, cmap=cmap_g2r, marker='o', s=8, alpha=1, zorder=4)
        plt.scatter(x, y, c=winds, marker='o', s=8, alpha=1, zorder=4)
        pos = data[advisnum]['lin']['points'][0]
        xh, yh = m(pos[0], pos[1])
        plt.plot(xh, yh, 'ok', markersize=2.5, alpha=0.5)
        shp_info_track = m.readshapefile(v['lin']['filename'].replace('.shp', ''), 'track', linewidth=2.5, color='k', zorder=5)
        shp_info_pgn = m.readshapefile(v['pgn']['filename'].replace('.shp', ''), 'cone', linewidth=0.0, color='b')
        patches=[]
        for info, shape in zip(m.cone_info, m.cone):
            patches.append(Polygon(np.array(shape), True))
        ax.add_collection(PatchCollection(patches, facecolor='b', alpha=0.25, edgecolor='b', linewidths=1., zorder=2))

        plt.annotate(v['date'], xy=(.5, -.1), xycoords='axes fraction', horizontalalignment='center')
        plt.annotate(f"advisory {advisnum}", xy=(.5, -.14), xycoords='axes fraction', horizontalalignment='center')
        plt.annotate(f"data: NOAA/NHC viz:{author}", xy=(1.27, -.127),
                     xycoords='axes fraction',
                     horizontalalignment='right', fontsize=6)
        plt.title(f"{v['STORMNAME']} forecasted track")
        # clb = plt.colorbar()
        # clb.set_label('winds (mph)')#, labelpad=-40, y=1.05, rotation=0)

        plt.savefig(pltfilename, format='png')
        plt.clf()

def previous_advisories(advisnum, data):
    pastadvs = []
    for advisnum2 in tqdm(data.keys()):
        if advisnum2 < advisnum:
            pastadvs.append(advisnum2)
    return pastadvs

def get_data_from_shapefiles(dir):
    basedir = f"{dir}/nhc_data"
    data = {}
    for dirname in sorted(os.listdir(basedir)):
        if dirname.startswith('.'):
            continue  # pesky .DS_Store files
        for filename in sorted(os.listdir(f"{basedir}/{dirname}")):
            if filename.endswith('.shp'):
                shpfilename = f"{basedir}/{dirname}/{filename}"
                shpf_track = shapefile.Reader(shpfilename)
                advisnum = shpf_track.records()[0].as_dict()['ADVISNUM']
                if 'A' in advisnum:
                    advisnum = "%03dA" % int(advisnum.replace('A', ''))
                else:
                    advisnum = "%03d" % int(advisnum)
                if advisnum not in data.keys():
                    data[advisnum] = {'date': None,
                                      'dt': None,
                                      'lin': None,
                                      'pts': None,
                                      'pgn': None,
                                      'lon_min': None,
                                      'lon_max': None,
                                      }

                for expectedshape in ['lin', 'pts', 'pgn']:
                    if shpfilename.endswith(f"_{expectedshape}.shp"):
                        shapes = shpf_track.shapes()
                        data[advisnum][expectedshape] = {'filename': shpfilename,
                                                         'shapes': shapes,
                                                         'points': shapes[0].points,
                                                         }
                        metadata = uniqueify_metadata(shpf_track.records())
                if 'ADVDATE' in metadata.keys():
                    metadata['date'], metadata['dt'] = format_date(metadata['ADVDATE'])
                data[advisnum].update(metadata)
                mins = min(data[advisnum]['lin']['points'])
                maxes = max(data[advisnum]['lin']['points'])
                data[advisnum]['lon_min'] = mins[0]
                data[advisnum]['lat_min'] = mins[1]
                data[advisnum]['lon_max'] = maxes[0]
                data[advisnum]['lat_max'] = maxes[1]
    return data

def uniqueify_metadata(records):
    metadata = dict()
    metadata_all = dict()
    for r in records:
        for k, v in r.as_dict().items():
            try:
                metadata_all[k].append(v)
            except:
                metadata_all[k] = [v]
    for k, v in metadata_all.items():
        if len(set(v)) == 1:
            # retain only metadata which applies across the entire advisory, drop anything that changes
            # with each point
            metadata[k] = v[0]
    return metadata

def makemovie(dir, duration=10, slowatend=True,maxfps=8):
    mypath = f"{dir}/frames"
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = []
    imgcnt=len(filenames)
    frameno = 0
    for filename in tqdm(sorted(filenames), unit='frame', desc='assemble frames'):
        frameno += 1
        accel = 0
        if slowatend:
            y = 1 / ((imgcnt +1) - frameno)
            r = round(maxfps*y)+1
            for x in range(0,r):
                images.append(imageio.imread(f"{mypath}/{filename}"))
        else:
            print (e)
            images.append(imageio.imread(f"{mypath}/{filename}"))
    fps = maxfps
    for ext in ['gif', 'mp4']:
        if 'gif' in ext:
            imageio.mimsave(f"{dir}/animations/{dir}_track_evolution.{ext}",
                            images, fps=fps, loop=1)
        else:
            imageio.mimsave(f"{dir}/animations/{dir}_track_evolution.{ext}",
                            images, fps=fps)

def format_date(date):
    dt = datetime.datetime.strptime(date.replace('AST', 'EST'), '%I%M %p %Z %a %b %d %Y')
    etzone = pytz.timezone('America/New_York')
    if 'AST' in date:
        # dt = dt - datetime.timedelta(hours=1)
        thetz = pytz.timezone('America/Santo_Domingo')
    if 'EDT' in date or 'EST' in date:
        thetz = pytz.timezone('America/New_York')
    dt = thetz.localize(dt)
    # dtet = dt.astimezone(pytz.timezone('America/NewYork'))
    dtet = dt.astimezone(pytz.timezone('America/New_York'))
    dtstr = dtet.strftime('%a %b %-d, %-I %p ') + dtet.tzname()
    dtstrshort = dtet.strftime('%a %-I%p')
    return dtstr, dt


class MyTestCase(unittest.TestCase):
    def test_log(self):
        accel = 0
        fps = 8.0
        for frameno in range(1,81):
            y = 1 / (81 - frameno)
            r = round(fps*y)+1
            # y = fps - y
            print (f"{frameno:2} {y:5.2} {r:5}")

    def test_3_movie(self):
        makemovie('2019Dorian')

    def test_dirs(self):
        for basedir in ['2019Dorian', '2018Florence']:
            setup_directories(basedir)
            for dir in ['animations', 'frames', 'nhc_data']:
               self.assertTrue(os.path.isdir(f"{basedir}/{dir}"))

    def test_2_Dorian(self):
        # get_advisories('2019Dorian', year=2019, storm=5)
        make_frames('2019Dorian', year=2019, storm=5, fetch_latest_adv=True)
        makemovie('2019Dorian')

    def test_3_Florence(self):
        # get_advisories('2018Florence', 2018, 6)
        make_frames('2018Florence', 2018, 6)
        # makemovie('2018Florence')
if __name__ == '__main__':
    make_frames('2019Dorian', year=2019, storm=5, fetch_latest_adv=True)
    makemovie('2019Dorian')
