import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from core.rasters import inspect_folder, coverage_message, preview, display_values

class RasterTests(unittest.TestCase):
    def test_folder_pair_coverage_and_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            for component in ('HH','HV'):
                path = Path(folder)/f'N16E080_2024_sl_{component}_F02DAR.tif'
                with rasterio.open(path,'w',driver='GTiff',height=40,width=40,
                                   count=1,dtype='uint16',crs='EPSG:4326',
                                   transform=from_origin(80,16,0.025,0.025),nodata=0) as dst:
                    data = np.full((40,40),100,dtype='uint16')
                    data[0,0] = 0
                    dst.write(data,1)
            records, errors = inspect_folder(folder)
            self.assertEqual(errors,[])
            self.assertEqual(len(records),2)
            inside = [{'geometry':mapping(box(80.1,15.1,80.2,15.2))}]
            outside = [{'geometry':mapping(box(79,15,80.2,15.2))}]
            self.assertTrue(coverage_message(inside,records,2024)[0])
            self.assertFalse(coverage_message(outside,records,2024)[0])
            self.assertFalse(coverage_message(inside,records,2025)[0])
            data,bounds = preview(records[0]['path'])
            self.assertLessEqual(max(data.shape),700)
            self.assertTrue(np.ma.getmaskarray(data)[0,0])
            self.assertAlmostEqual(float(display_values(data,True).mean()),-43)

    def test_projected_raster_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'N16E080_2024_sl_HH_F02DAR.tif'
            with rasterio.open(path,'w',driver='GTiff',height=10,width=10,
                               count=1,dtype='uint16',crs='EPSG:32644',
                               transform=from_origin(400000,1700000,25,25)) as dst:
                dst.write(np.ones((10,10),dtype='uint16'),1)
            records,errors = inspect_folder(folder)
            self.assertFalse(errors)
            self.assertTrue(79 < records[0]['bounds'][0] < 82)
            data,bounds = preview(path)
            self.assertTrue(79 < bounds[0] < 82)

if __name__ == '__main__':
    unittest.main()
