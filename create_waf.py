'''
This creates Northern Hemisphere WAFx, Wafy, Wafz, one file per day,
for a range of dates, using ERA5 data and Eq. 7.1 from Plumb 1985

 - input:   None
 - output:  Netcdf file, e.g. wafx_20210201.nc
            Folder structure    ---waf
                                  |--wafx
                                  |--wafy
                                  |--wafz

Thanks to Jason Furtado for examples of code to compute wafz and the
stability parameter


'''

import pandas as pd
import numpy as np
import xarray as xr
import cdsapi
import os
import tempfile

############ Settings ##########################################################
#
# Adjust these
#
sdate='20210101'  #start date of range to process, format 'yyyymmdd'
edate='20210101'  #end date of range to process, format 'yyyymmdd'
dowafx=True       #if True, creates wafz for date range specified
dowafy=True       #if True, creates wafy for date range specified
dowafz=True       #if True, creates wafz for date range specified
overwrite=False   #set this to True if you want to recreate WAF files, otherwise
                  #only missing files in date range are created
outdir='waf'      #top level of folder structure to store WAF files
storedata=True    #if False, data to produce WAF is downloadeded first from Copernicus
                  #(this is time-consuming but uses no permanent disk-space)
                  #if True, get data from directory <datadir>:
                  #data is expected as a single era5 file for each day, 20-90N only,
                  #daily mean, 4 variables, all levels: u, v, t, z
                  #if True but file does not exist, the data is downloaded and saved
datadir='era5_dm' #if storedata==True, set to location to find or create file 'data_<yyyymmdd>.nc'

################################################################################

## helpful functions
#this function is used for stability measure
def areal_avg(fld,lats,latdim=0,londim=1):
    """
    Expecting fld (N-D) with lat,lon dimensions
    The field is meaned over the lat/lons with weighting for lats
    Returns (N-2)-D field
    Avoids masked data, and handles nans
    """
    #quick error check, fld must be at least 2 dimensions, latdim, londim must be reasonable
    fldshape=fld.shape
    flddim=fld.ndim
    if flddim<2 or lats.ndim!=1 or len(lats)!=fldshape[latdim]:
        print('areal_avg error: dimension mismatch')
        return(np.NaN)

    #create weight field, expand dims to match input
    wt = np.cos(np.radians(lats))
    for i in range(0,latdim): #pad beginning
        wt=np.expand_dims(wt,0)
    for i in range(latdim+1,flddim): #pad end
        wt=np.expand_dims(wt,-1)

    a=np.nanmean(fld,axis=londim,keepdims=True)
    aa=np.squeeze(np.nansum(a*wt/np.sum(wt),axis=latdim))

    return aa

#This function performs the download from Copernicus of the data used to create
#WAF (u,v,t,z). Can be very time consuming.
def download_data(tdate,filen):
    '''
    Downloads era5 data from copernicus data server using cdsapi

    Input
    tdate: date to retrieve, as a character string 'yyyymmdd'
    filen: save to <filename>

    Output
    netcdf file <filen>
    contains ERA5 u,v,t,z for Northern Hemisphere 20-90N, all levels, all lons, daily mean
    returns True if all is good
    '''

    print(f'{tdate}...downloading data')

    c = cdsapi.Client() #start new client
    year=tdate[0:4]
    mon=tdate[4:6]
    day=tdate[6:8]
    c.retrieve("reanalysis-era5-pressure-levels",
    {
    'product_type': 'reanalysis',
    'format': 'netcdf',
    'variable': [
        'geopotential', 'temperature', 'u_component_of_wind',
        'v_component_of_wind',
        ],
    'year': year,
    'month': mon,
    'day': day,
    'time': [
        '00:00', '01:00', '02:00',
        '03:00', '04:00', '05:00',
        '06:00', '07:00', '08:00',
        '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00',
        '15:00', '16:00', '17:00',
        '18:00', '19:00', '20:00',
        '21:00', '22:00', '23:00',
        ],
    'pressure_level': [
        '1', '2', '3',
        '5', '7', '10',
        '20', '30', '50',
        '70', '100', '125',
        '150', '175', '200',
        '225', '250', '300',
        '350', '400', '450',
        '500', '550', '600',
        '650', '700', '750',
        '775', '800', '825',
        '850', '875', '900',
        '925', '950', '975',
        '1000',
        ],
    'area': [
        90, -180, 20,
        180,
        ],
    },filen)

    #make this a daily mean and save
    ds=xr.open_dataset(filen)
    ds=ds.resample(time='D').mean(dim='time',skipna=True)
    ds.to_netcdf(filen) #rewrites as daily mean

    return True

#####  main code
#create output folders
if storedata==True and not os.path.isdir(datadir):
    os.makedirs(datadir,exist_ok=True)
if dowafx==True:
    os.makedirs(f'{outdir}/wafx',exist_ok=True)
if dowafy==True:
    os.makedirs(f'{outdir}/wafy',exist_ok=True)
if dowafz==True:
    os.makedirs(f'{outdir}/wafz',exist_ok=True)

#list of dates to process
pdates=pd.date_range(start=sdate,end=edate,freq='D')
dates=pdates.strftime('%Y%m%d').values
#idates=pdates.strftime('%Y%m%d').values.astype(int)

#loop through dates, downloading data if necessary, and processing
for idx,tdate in enumerate(dates):

    with tempfile.TemporaryDirectory() as tmpdirname: #used if we need to download

        if storedata==False: #always download
            infile=f'{tmpdirname}/data_{tdate}.nc'
            res=download_data(tdate,infile)
            data=xr.load_dataset(infile)
        else: #only download if not already stored
            infile=f'{datadir}/data_{tdate}.nc'
            if not os.path.isfile(infile):
                infile2=f'{tmpdirname}/data_{tdate}.nc' #change location
                res=download_data(tdate,infile2)
                data=xr.load_dataset(infile2)
                data.to_netcdf(infile) #store it
            else:
                data=xr.load_dataset(infile)
                #can do some data checking
                #for correct latitude range
                #single element in time dimension
                #u,v,t,z all present


        #get grid coordinates, use numpy arrays when convenient
        lats=data.latitude
        lons=data.longitude
        levs=data.level

        lats0=lats.to_numpy()
        lons0=lons.to_numpy()
        levs0=levs.to_numpy()
        rlat=np.radians(lats0)
        rlon=np.radians(lons0)

        #set up with some constants and other fields
        gfactor=1.0  ##make this 1.0 if the z field is geopotential, 9.806 if geopotential height

        #some constants, make sure g is set to 1.0 if z is already in geopotential
        a = 6.3781e6
        omega = 7.29e-5
        R = 287.05; cp = 1004.; kappa = R/cp
        g = 9.806; To = 260.; H = R*To/g

        #some values for WAF calc, Plumb 1985
        fcor=2*omega*np.sin(rlat)
        C=2*omega*a*np.sin(2*rlat)
        coslat=np.cos(rlat)
        plevs=levs0/1000 #this is 'p' in Plumb 1985

        #expand some dimensions for easy broadcasting
        C3=np.expand_dims(np.expand_dims(C,0),-1)        #1 x lats x 1
        coslat3=np.expand_dims(np.expand_dims(coslat,0),-1) #1 x lats x 1
        plevs3=np.expand_dims(np.expand_dims(plevs,-1),-1) #levs x 1 x 1
        fcor3=np.expand_dims(np.expand_dims(fcor,0),-1)  #1 x lats x 1

        #wafx
        #uses numpy for calculations - this can be switched to xarray functions
        if dowafx==True:
            outfile=f'{outdir}/wafx/wafx_{tdate}.nc'
            if overwrite==True or not os.path.isfile(outfile):
                print(f'{tdate}...wafx')
                #get V
                V=data.v.squeeze() #eliminate time dimension to get (lev,lat,lon)
                #get Z (phi -- geopotential)
                Z=data.z.squeeze()*gfactor
                #get V'V'
                VVflux=(V-np.nanmean(V,axis=2,keepdims=True))*(V-np.nanmean(V,axis=2,keepdims=True)) #(lev,lat,lon)
                #get V'Z'
                VZflux=(V-np.nanmean(V,axis=2,keepdims=True))*(Z-np.nanmean(Z,axis=2,keepdims=True)) #(lev,lat,lon)
                #get d(V'Z')/d(lamba)
                dVZ_dlon = np.gradient(VZflux,axis=2,edge_order=2) #(lev,lat,lon)
                #Plumb 1985 Eq. 7.1 for WAFx
                tmp2=plevs3*coslat3*(VVflux-(dVZ_dlon/C3))  #(lev,lat,lon)
                #save as netcdf
                da=xr.DataArray(data=tmp2,
                        name='wafx',
                        dims=['level','latitude','longitude'],
                        coords={'time':pdates[idx],'level':levs,'latitude':lats,'longitude':lons},
                        attrs={'description':'x-component of waf','units':'m2s-2'}) #attributes here
                da.to_netcdf(outfile)

        #wafy
        if dowafy==True:
            outfile=f'{outdir}/wafy/wafy_{tdate}.nc'
            if overwrite==True or not os.path.isfile(outfile):
                print(f'{tdate}...wafy')
                #get U
                U=data.u.squeeze()
                #get V
                V=data.v.squeeze()
                #get Z (phi -- geopotential)
                Z=data.z.squeeze()*gfactor
                #get U'V'
                UVflux=(U-np.nanmean(U,axis=2,keepdims=True))*(V-np.nanmean(V,axis=2,keepdims=True)) #(lev,lat,lon)
                #get U'Z'
                UZflux=(U-np.nanmean(U,axis=2,keepdims=True))*(Z-np.nanmean(Z,axis=2,keepdims=True)) #(lev,lat,lon)
                #get d(U'Z')/d(lamba)
                dUZ_dlon = np.gradient(UZflux,axis=2,edge_order=2) #(lev,lat,lon)
                #Plumb 1985 Eq. 7.1 for WAFx
                tmp2=plevs3*coslat3*((-1)*UVflux+(dUZ_dlon/C3))  #(lev,lat,lon)
                #save as netcdf
                da=xr.DataArray(data=tmp2,
                        name='wafy',
                        dims=['level','latitude','longitude'],
                        coords={'time':pdates[idx],'level':levs,'latitude':lats,'longitude':lons},
                        attrs={'description':'y-component of waf','units':'m2s-2'}) #attributes here
                da.to_netcdf(outfile)


        #wafz
        if dowafz==True:
            outfile=f'{outdir}/wafz/wafz_{tdate}.nc'
            if overwrite==True or not os.path.isfile(outfile):
                print(f'{tdate}...wafz')
                #get T
                T=data.t.squeeze()
                #get V
                V=data.v.squeeze()
                #get Z (phi -- geopotential)
                Z=data.z.squeeze()*gfactor
                #get stability for this, use exponential atm assumptions
                Tmean=areal_avg(T,lats0,1,2)  #(p,)
                dz = -H*np.log(plevs)  #from Plumb 1985 p.218  (p,)
                dTmean_dz=np.gradient(Tmean,dz) #(p,)
                stab=dTmean_dz+(kappa*Tmean/H)  #from Plumb 1885 p. 218, eq 2.2
                stab3=np.expand_dims(np.expand_dims(stab,-1),-1) #levs x 1 x 1
                #get v'T'
                VTflux=(V-np.nanmean(V,axis=2,keepdims=True))*(T-np.nanmean(T,axis=2,keepdims=True))
                #get T'Z'
                TZflux=(T-np.nanmean(T,axis=2,keepdims=True))*(Z-np.nanmean(Z,axis=2,keepdims=True))
                #get d(ZT)/d(lamba)
                dTZ_dlon = np.gradient(TZflux,axis=2,edge_order=2)
                #Plumb 1985 Eq. 7.1 for WAFz
                tmp2 = plevs3*coslat3*(fcor3/stab3*(VTflux-dTZ_dlon/C3))  #lev,lat,lon
                #save as netcdf
                da=xr.DataArray(data=tmp2,
                        name='wafz',
                        dims=['level','latitude','longitude'],
                        coords={'time':pdates[idx],'level':levs,'latitude':lats,'longitude':lons},
                        attrs={'description':'z-component of waf','units':'m2s-2'}) #attributes here
                da.to_netcdf(outfile)

