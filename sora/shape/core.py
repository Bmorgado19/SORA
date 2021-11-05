import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation

from .meta import BaseShape
from .utils import read_obj_file, read_obj_file2 

__all__ = ['Shape3D']


class Shape3D(BaseShape):
    """Defines a class to handle a 3D shape object.

    Parameters
    ----------
    obj_file : `str`
        Path to the Wavefront OBJ file.

    texture : `str`
        Path to the image that contains the surface texture of the object.
        If None, then a gray color will be defined.
    """

    def __init__(self, obj_file, texture=None, scale=1) -> None:
        super(Shape3D, self).__init__()
        try:
            vertices, faces = read_obj_file(obj_file)
        except:
            vertices, faces = read_obj_file2(obj_file)
        self.name = obj_file
        self._vertices = CartesianRepresentation(*vertices.T, unit=u.km)
        self.scale = scale
        if faces.min() == 1:
            faces = faces - 1
        self.faces = faces
        if texture is None:
            self.texture = 0.5 * np.ones((len(faces), 3))
        else:
            import matplotlib.pyplot as plt
            img = plt.imread(texture)
            a, b = img.shape[0:2]
            ym, xm = np.indices(img.shape[0:2])
            yt = (ym + 0.5 - a / 2.0) / (a + 1)
            xt = (xm + 0.5 - b / 2.0) / (b + 1)
            k = np.where(xt < 0.0)
            xt[k] = xt[k] + 1.0

            long_img = -xt * 360 * u.deg + 360 * u.deg
            lat_img = -yt * 180 * u.deg

            img_lats = SkyCoord(ra=-long_img.flatten(), dec=lat_img.flatten())
            faces = SkyCoord(self.vertices[self.faces].mean(axis=-1))

            idx, d2d, d3d = faces.match_to_catalog_sky(img_lats)
            texture = img[ym.flatten()[idx], xm.flatten()[idx]]
            self.texture = np.array(texture) / 256.0

    @property
    def vertices(self):
        return self._vertices * self.scale

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = float(value)

    def rotated_vertices(self, sub_observer="00 00 00 +00 00 00", pole_position_angle=0):
        """Returns the vertices rotated as viewed by a given observer,
        with 'x' in the direction of the observer.

        Parameters
        ----------
        sub_observer : `astropy.coordinates.SkyCoord`, `str`
            Planetocentric coordinates of the center of the object as seen by the observer.
            It can be an astropy SkyCoord object or a string with the bodycentric longitude
            latitude in degrees. Ex: "30.0 -20.0", or "30 00 00 -20 00 00".

        pole_position_angle : `float`, `int`
            Body's North Pole position angle with respect to direction of the ICRS
            North Pole, i.e. N-E-S-W.

        Returns
        -------
        rotated_vertices : `astropy.coordinates.CartesianRepresentation`
            An astropy object with the rotated vertices as viewed by the observer.
            The 'x' coordinate are in the direction of the observer, the 'z' coordinate
            in the direction of the projected ICRS North Pole, and the 'y' coordinate
            complementing the right-hand rule.
        """
        from astropy.coordinates.matrix_utilities import rotation_matrix

        if isinstance(sub_observer, str):
            sub_observer = SkyCoord(sub_observer, unit=(u.deg, u.deg))
        sub_observer = sub_observer.spherical

        pa = u.Quantity(pole_position_angle, unit=u.deg)
        rz = rotation_matrix(-sub_observer.lon, axis='z')
        ry = rotation_matrix(-sub_observer.lat, axis='y')
        rx = rotation_matrix(pa, axis='x')
        rotated_vertices = self.vertices.transform(rz).transform(ry).transform(rx)
        return rotated_vertices

    def plot(self, sub_observer="00 00 00 +00 00 00", sub_solar=None, pole_position_angle=0, center_f=0, center_g=0,
             radial_offset=0, ax=None):
        """

        Parameters
        ----------
        sub_observer : `astropy.coordinates.SkyCoord`, `str`
            Planetocentric coordinates of the center of the object in the direction of the observer.
            It can be an astropy SkyCoord object or a string with the bodycentric longitude
            latitude in degrees. Ex: "30.0 -20.0", or "30 00 00 -20 00 00".

        sub_solar : `astropy.coordinates.SkyCoord`, `str`
            Planetocentric coordinates of the center of the object in the direction of the Sun.
            It can be an astropy SkyCoord object or a string with the bodycentric longitude
            latitude in degrees. Ex: "30.0 -20.0", or "30 00 00 -20 00 00".

        pole_position_angle : `float`, `int`
            Body's North Pole position angle with respect to direction of the ICRS
            North Pole, i.e. N-E-S-W.

        center_f : `int`, `float`
            Offset of the center of the body in the East direction, in km

        center_g  : `int`, `float`
            Offset of the center of the body in the North direction, in km

        radial_offset : `int`, `float`
            Offset of the center of the body in the direction of observation, in km

        ax : `matplotlib.pyplot.Axes`
            The axes where to make the plot. If None, it will use the default axes.
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.axis('equal')

        vertices_observer = self.rotated_vertices(sub_observer=sub_observer, pole_position_angle=pole_position_angle)
        cart_obs = vertices_observer[self.faces.T]
        ab = cart_obs[0] - cart_obs[1]
        ac = cart_obs[0] - cart_obs[-1]
        normal_obs = ab.cross(ac)
        normal_obs = normal_obs / normal_obs.norm()

        observable = normal_obs.x >= 0

        if not sub_solar:
            sub_solar = sub_observer
        vertices_sun = self.rotated_vertices(sub_observer=sub_solar)

        cart_sun = vertices_sun[self.faces.T]
        ab = cart_sun[0] - cart_sun[1]
        ac = cart_sun[0] - cart_sun[2]
        normal_sun = ab.cross(ac)
        normal_sun = normal_sun / normal_sun.norm()

        shade = normal_sun.x
        shade[shade < 0] = 0
        alpha = np.ones(len(shade))
        color = np.vstack((self.texture.T * shade, alpha)).T

        for i, pol in enumerate(cart_obs.T):
            if not observable[i]:
                continue
            ax.fill(-pol.y.value + center_f, pol.z.value + center_g, color=color[i],
                    zorder=pol.x.value.mean() - radial_offset)

    def get_limb(self, sub_observer="00 00 00 +00 00 00", pole_position_angle=0):
        from astropy.coordinates import Angle

        rv = self.rotated_vertices(sub_observer=sub_observer, pole_position_angle=pole_position_angle)
        k = np.repeat(False, len(rv))
        for i, v in enumerate(rv):
            kk = np.repeat(True, len(rv))
            kk[i] = False
            angs = Angle(np.arctan2(rv[kk].z - v.z, rv[kk].y - v.y))
            angs = angs.wrap_at(360 * u.deg)
            angs.sort()
            if np.absolute(angs[1:] - angs[:-1]).max() > 120 * u.deg or angs[-1] - angs[0] < 240 * u.deg:
                k[i] = True
        return -rv[k].y, rv[k].z

    def get_limb2(self, sub_observer="00 00 00 +00 00 00", pole_position_angle=0):
        from astropy.coordinates import Angle

        rv = self.rotated_vertices(sub_observer=sub_observer, pole_position_angle=pole_position_angle)
        #mindist = (rv[self.faces.T[0]] - rv[self.faces.T[1]]).norm().max()
        #start = np.argmax(np.sqrt(rv.y*2 + rv.z*2))
        #k = np.repeat(False, len(rv))
        #k[start] = True
        #current = start
        #former = None
        cart_obs = rv[self.faces.T]
        ab = cart_obs[0] - cart_obs[1]
        ac = cart_obs[0] - cart_obs[-1]
        normal_obs = ab.cross(ac)
        normal_obs = normal_obs / normal_obs.norm()
        k = np.where(np.absolute(normal_obs.x) < np.sin(5 * u.deg))
        vert = np.unique(self.faces[k].flatten())
        # while True:
        #     v = rv[current]
        #     close = np.where(np.sqrt((rv.z - v.z*2) + (rv.y - v.y)*2) < 5*mindist)[0]
        #     kk = np.repeat(True, len(close))
        #     kk[current] = False
        #     angs = Angle(np.arctan2(rv[kk].z - v.z, rv[kk].y - v.y))
        #     angs = angs.wrap_at(360 * u.deg)
        #     angs.sort()
        #
        #     break
        k = np.repeat(False, len(vert))
        for i, v in enumerate(rv[vert]):
            kk = np.repeat(True, len(vert))
            kk[i] = False
            angs = Angle(np.arctan2(rv[vert][kk].z - v.z, rv[vert][kk].y - v.y))
            angs = angs.wrap_at(360 * u.deg)
            angs.sort()
            if np.absolute(angs[1:] - angs[:-1]).max() > 120 * u.deg or angs[-1] - angs[0] < 240 * u.deg:
                k[i] = True
        return -rv[vert][k].y, rv[vert][k].z
        # return -vertices_observer[vert].y, vertices_observer[vert].z