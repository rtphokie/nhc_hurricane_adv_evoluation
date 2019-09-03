import unittest
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import zipfile
from io import StringIO
import warnings
import requests
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from tqdm import tqdm
import os,io
import shapefile
import datetime
import pytz
from os import listdir
from os.path import isfile, join
import logging
from pprint import pprint
'''
https://www.nhc.noaa.gov/gis/
'''
import matplotlib.colors as mcolors

cdict = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 0.0, 0.0))}

cmap_g2r = mcolors.LinearSegmentedColormap( 'my_colormap', cdict, 200)


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

def get_advisories(dir, pref):
    #https://www.nhc.noaa.gov/gis/forecast/archive/al052019_5day_036.zip
    done = False
    goget = []
    for filename in sorted(os.listdir(dir)):
        advisory = filename.replace('al052019_5day_', '')
    if 'A' not in advisory:
        goget.append("%03dA" % int(advisory))
    goget.append("%03d" % (int(advisory.replace('A', '')) + 1))
    statuses=[]
    for s in goget:
        zip_file_url = f"https://www.nhc.noaa.gov/gis/forecast/archive/{pref}_{s}.zip"
        r = requests.get(zip_file_url, stream=True)
        statuses.append(r.status_code)
        if r.status_code < 300:
            cwd = os.getcwd()
            path = f"{dir}/{pref}_{s}"
            os.mkdir(path)
            print (f"created {path}")
            os.chdir(path)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()
            os.chdir(cwd)

    if not list(set(statuses))[0] >= 300:
        # keep going until all downloads fail
        get_advisories(dir=dir, pref=pref)

def make_frames(dir, adv, fetch_new_advisories=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    if fetch_new_advisories:
        get_advisories(dir, adv)

    data = get_data_from_shapefiles(dir)

    pastadvisories = []
    cmap = plt.cm.get_cmap('binary')
    for advisnum, v in tqdm(sorted(data.items())):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pltfilename = 'frames/themap_%s.png' % advisnum
        if os.path.exists(pltfilename):
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
        plt.scatter(x, y, c=winds, cmap=cmap_g2r, marker='o', s=8, alpha=1, zorder=4)
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
        plt.annotate(f"data: NOAA/NHC viz:Tony Rice @rtphokie", xy=(1.27, -.127),
                     xycoords='axes fraction',
                     horizontalalignment='right', fontsize=6)
        plt.title(f"{v['STORMNAME']} forecasted track")
        # clb = plt.colorbar()
        # clb.set_label('winds (mph)')#, labelpad=-40, y=1.05, rotation=0)

        plt.savefig(pltfilename)
        pastadvisories.append(advisnum)
        # pprint(v)
        plt.clf()


def previous_advisories(advisnum, data):
    pastadvs = []
    for advisnum2 in tqdm(data.keys()):
        if advisnum2 < advisnum:
            pastadvs.append(advisnum2)
    return pastadvs


def get_data_from_shapefiles(dir):
    data = {}
    for dirname in sorted(os.listdir(dir)):
        if dirname.startswith('.'):
            continue
        for filename in sorted(os.listdir(f"{dir}/{dirname}")):
            if filename.endswith('.shp'):
                shpfilename = f"{dir}/{dirname}/{filename}"
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

def makemovie(mypath):
    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    import imageio
    images = []
    for filename in tqdm(sorted(filenames), unit='frame', desc='assemble frames'):
        try:
            images.append(imageio.imread(f"{mypath}/{filename}"))
        except:
            print(f"skipping {filename}")
    imageio.mimsave('dorian_track_evolution.gif', images, fps=2.5)
    imageio.mimsave('dorian_track_evolution.mp4', images, fps=2)

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
    def test_3_movie(self):
        makemovie('frames')

    def test_0_plot(self):
        positions = []
        dtss = []
        positions, dtss = main("%03d" % 1, positions, dtss)
        positions, dtss = main("%03d" % 15, positions, dtss)
        positions, dtss = main("%03d" % 25, positions, dtss)
        positions, dtss = main("%03dA" % 36, positions, dtss)

    def test_1_plots(self):
        get_advisories('2019Dorian', 'al052019_5day')
        positions = []
        dtss = []
        for filename in tqdm(sorted(os.listdir('2019Dorian'))):
            if '_5day_' in filename:
                advisory = filename.replace('al052019_5day_','')
                positions, dtss = main(advisory, positions, dtss)

    def test_0_ad(self):
        get_advisories()

if __name__ == '__main__':
    make_frames('2019Dorian', 'al052019_5day', fetch_new_advisories=True)
    makemovie('frames')
