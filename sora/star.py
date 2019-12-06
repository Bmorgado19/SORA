from astropy.coordinates import SkyCoord, SphericalCosLatDifferential
from astropy.time import Time
import astropy.units as u
from astroquery.vizier import Vizier
from .config import test_attr

### Object for star
def search_star(coord, columns, radius, catalog):
    """ Search position on Vizier and returns a catalogue.

    Parameters:
        coord (str, astropy.SkyCoord):Coordinate to search aroung.
        columns (list):List of strings with the name of the columns to retrieve.
        radius (int, float, unit.quantity):Radius to search around coord.
        catalog (str):Vizier catalogue to search.

    Returns:
        catalogue(astropy.Table):An astropy Table with the catalogue informations.   

    """
    row_limit = 100
    print('Downloading coordinate positions around {}'.format(radius))
    vquery = Vizier(columns=columns, row_limit=row_limit, timeout=600)
    catalogue = vquery.query_region(coord, radius=radius, catalog=catalog)
    return catalogue

class Star():
    '''
    Docstring
    Define a star
    '''
    def __init__(self, **kwargs):
        # Run initial parameters
        #if 'coord' in kwargs:
        #    self.star = test_attr(kwargs['coord'], SkyCoord, 'coord')
        #elif 'code' in kwargs:
        #    self.star = test_attr(kwargs['code'], str, 'code')
        #else:
        #    raise KeyError('Input values must have either the coordinate of the star or the Gaia code for online search')
        #if type(self.star) != SkyCoord:
        #    pos = self.search_gaia2()
        return
    
    def apparent_radius(self, distance):
        # calculate the apparent radius of the star at given distance
        return
    
    def __searchgaia2(self):
        # search for the star position in the gaia catalogue and save informations
        #columns = ['Source', 'RA_ICRS', 'e_RA_ICRS', 'DE_ICRS', 'e_DE_ICRS', 'Plx', 'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE', 'Gmag', 'e_Gmag', 'Dup', 'Epoch', 'Rad']
        #catalogue = search_star(coord=self.star, columns=columns, radius=2*u.arcsec, catalog='I/345/gaia2')
        return
        
    def __getcolors(self):
        # search for the B,V,K magnitudes of the star on Vizier and saves the result
        return
        
    def geocentric(self, time):
        # calculate the position of the star using parallax and proper motion
        return
    
    def barycentric(self, time):
        # calculate the position of the star using proper motion
        return
    
    def add_offset(self, da_cosdec, ddec):
        # saves an offset for the star
        
        #dadc = test_attr(da_cosdec, u.quantity.Quantity, 'd_lon_coslat')
        #dd = test_attr(ddec, u.quantity.Quantity, 'dd')
        #self.delta = SphericalCosLatDifferential(dadc, dd, 0.0*u.km)
        return
        
    
    def __str__(self):
        # return what it is to be printed
        return ''