import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest
from core.rasters import inspect_folder
from settings import ROOT

class DashboardTests(unittest.TestCase):
    def test_loaded_raster_renders_and_db_toggle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'N16E080_2024_sl_HH_F02DAR.tif'
            with rasterio.open(path,'w',driver='GTiff',height=20,width=20,
                               count=1,dtype='uint16',crs='EPSG:4326',
                               transform=from_origin(80,16,.05,.05),nodata=0) as dst:
                dst.write(np.arange(400,dtype='uint16').reshape(20,20),1)
            records,_ = inspect_folder(folder)
            app = AppTest.from_file(str(ROOT/'app.py'))
            app.session_state['records'] = records
            app.session_state['loaded_folder'] = folder
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric),3)
            app.checkbox[0].check().run(timeout=30)
            self.assertFalse(app.exception)

if __name__ == '__main__':
    unittest.main()
