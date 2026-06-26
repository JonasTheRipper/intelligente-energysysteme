-- Enable PostGIS in the wildfire database on first container start.
-- (The wildfire_cma.postgis module also creates the extension idempotently,
--  but enabling it here means the DB is GIS-ready the moment it boots.)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
