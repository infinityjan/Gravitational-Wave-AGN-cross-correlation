#!/usr/bin/env python4
"""
GLADE+ × GW Skymap 3D Cross-Match - FINAL VERIFIED VERSION
===========================================================
Properly classifies Galaxies (G) and Quasars (Q) in 90% credible region
"""

import sys
import numpy as np
import warnings
import h5py
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional
import time
import traceback

from astropy.table import Table
import astropy.units as u
import healpy as hp
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import find_greedy_credible_levels

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ========================================================================
# CHECKPOINT MANAGEMENT
# ========================================================================

class CrossmatchCheckpoint:
    """Manage cross-match checkpoints for resumable processing."""
    
    def __init__(self, output_fits: str):
        self.checkpoint_file = Path(f".{Path(output_fits).stem}_xmatch_checkpoint.json")
    
    def save(self, state: dict) -> None:
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self) -> Optional[dict]:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def delete(self) -> None:
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


# ========================================================================
# CATALOG FORMAT DETECTION
# ========================================================================

def detect_catalog_format(catalog_file: str) -> str:
    """Detect if catalog is HDF5 or ASCII."""
    catalog_path = Path(catalog_file)
    
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_file}")
    
    suffix = catalog_path.suffix.lower()
    if suffix in ['.h5', '.hdf5']:
        return 'hdf5'
    elif suffix == '.txt':
        return 'ascii'
    
    try:
        with open(catalog_file, 'rb') as f:
            magic = f.read(8)
        return 'hdf5' if magic.startswith(b'\x89HDF') else 'ascii'
    except Exception:
        return 'ascii'


# ========================================================================
# GW SKYMAP LOADING
# ========================================================================

class GWSkymapLoader:
    """Load and process GW skymaps."""
    
    def __init__(self, skymap_fits: str, verbose: bool = True):
        self.skymap_fits = skymap_fits
        self.verbose = verbose
        self.skymap_data = None
        self.metadata = {}
        self.is_3d = False
        self.is_moc = False
        self._load()
    
    def _load(self):
        if self.verbose:
            print(f"[1/5] Loading GW skymap from: {self.skymap_fits}")
        
        try:
            self.skymap_data, self.metadata = read_sky_map(self.skymap_fits, moc=True)
            self.is_moc = True
            if self.verbose:
                print("     ✓ Loaded as multi-order (MOC) format")
        except Exception:
            try:
                self.skymap_data, self.metadata = read_sky_map(self.skymap_fits, moc=False)
                self.is_moc = False
                if self.verbose:
                    print("     ✓ Loaded as standard HEALPix format")
            except Exception as e:
                raise RuntimeError(f"Failed to load skymap: {e}")
        
        self.is_3d = self._check_3d()
        if self.verbose:
            dim_str = "3D (with distance)" if self.is_3d else "2D (sky only)"
            print(f"     ✓ Skymap type: {dim_str}")
    
    def _check_3d(self) -> bool:
        if self.is_moc:
            return 'DISTMU' in self.skymap_data.colnames
        else:
            if isinstance(self.skymap_data, np.ndarray) and self.skymap_data.dtype.names:
                return 'DISTMU' in self.skymap_data.dtype.names
        return False
    
    def get_prob_array(self) -> np.ndarray:
        if self.is_moc:
            return np.asarray(self.skymap_data['PROBDENSITY'])
        else:
            if isinstance(self.skymap_data, np.ndarray) and self.skymap_data.dtype.names:
                return np.asarray(self.skymap_data['PROB'])
            else:
                return np.asarray(self.skymap_data)
    
    def get_distance_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_3d:
            raise ValueError("Skymap is 2D (no distance information)")
        
        if self.is_moc:
            distmu = np.asarray(self.skymap_data['DISTMU'])
            distsigma = np.asarray(self.skymap_data['DISTSIGMA'])
            distnorm = np.asarray(self.skymap_data['DISTNORM'])
        else:
            distmu = np.asarray(self.skymap_data['DISTMU'])
            distsigma = np.asarray(self.skymap_data['DISTSIGMA'])
            distnorm = np.asarray(self.skymap_data['DISTNORM'])
        
        return distmu, distsigma, distnorm
    
    def get_healpix_indices(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        if self.is_moc:
            import astropy_healpix as ahp
            uniq = self.skymap_data['UNIQ']
            levels, ipix = ahp.uniq_to_level_ipix(uniq)
            max_level = np.max(levels)
            max_nside = ahp.level_to_nside(max_level)
            theta = np.radians(90.0 - dec_deg)
            phi = np.radians(ra_deg)
            ipix_in = hp.ang2pix(max_nside, theta, phi, nest=True)
            return ipix_in
        else:
            prob = self.get_prob_array()
            npix = len(prob)
            nside = hp.npix2nside(npix)
            theta = np.radians(90.0 - dec_deg)
            phi = np.radians(ra_deg)
            ipix = hp.ang2pix(nside, theta, phi, nest=False)
            return ipix


# ========================================================================
# ASCII CATALOG STREAMING READER - VERIFIED
# ========================================================================

class ASCIICatalogReader:
    """Memory-efficient ASCII catalog reader with proper type classification."""
    
    def __init__(self, txt_file: str, chunk_size: int = 10000, verbose: bool = True):
        self.txt_file = txt_file
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        # Column definitions - Column 7 is type_code ('G' or 'Q')
        self.col_indices = [8, 9, 7, 28, 29, 10, 20, 32, 33, 35, 38]
   What We Need To Do:

Step 1 — For each of 86 GW events:

    Count how many quasars are in the 90% credible region → simple count
    Sum their W1 luminosities → weighted count
    This gives you 2 numbers per event
     self.col_names = [
            'RA', 'Dec', 'type_code', 'z_CMB', 'z_flag',
            'mag_B', 'mag_W1', 'DL', 'DL_err', 'mass', 'merger_rate'
        ]
        
        if verbose:
            print(f"[2/5] Counting ASCII lines...")
        
        self.total_lines = 0
        with open(txt_file, 'r') as f:
            for _ in f:
                self.total_lines += 1
        
        if verbose:
            print(f"     ✓ Total lines: {self.total_lines:,}")
    
    def iter_chunks(self):
        """Iterate over catalog in chunks, preserving type_code as STRING."""
        chunk_num = 0
        n_chunks = int(np.ceil(self.total_lines / self.chunk_size))
        
        with open(self.txt_file, 'r') as f:
            chunk_data = {col: [] for col in self.col_names}
            chunk_rows = 0
            
            for line_idx, line in enumerate(f):
                if line.strip() == '' or line.startswith('#'):
                    continue
                
                try:
                    parts = line.split()
                    
                    # CRITICAL: Extract columns with proper type handling
                    for col_idx, col_name in zip(self.col_indices, self.col_names):
                        try:
                            if col_name == 'type_code':
                                # KEEP AS STRING - don't convert to number!
                                val = parts[col_idx].strip()
                            else:
                                # All other columns are numeric
                                val = float(parts[col_idx])
                            
                            chunk_data[col_name].append(val)
                        except (IndexError, ValueError):
                            if col_name == 'type_code':
                                chunk_data[col_name].append('Unknown')
                            else:
                                chunk_data[col_name].append(np.nan)
                    
                    chunk_rows += 1
                
                except Exception:
                    continue
                
                # Yield chunk when buffer full
                if chunk_rows >= self.chunk_size:
                    df = pd.DataFrame(chunk_data)
                    
                    # CRITICAL: Classify based on STRING comparison
                    if 'type_code' in df.columns:
                        df['type'] = df['type_code'].apply(
                            lambda x: 'Galaxy' if str(x).strip().upper() == 'G' else 
                                     'Quasar' if str(x).strip().upper() == 'Q' else 
                                     'Unknown'
                        )
                    
                    # Verify classification on first chunk
                    if chunk_num == 0 and self.verbose:
                        print(f"\n     ✓ First chunk type distribution:")
                        type_counts = df['type'].value_counts()
                        for obj_type, count in type_counts.items():
                            pct = 100 * count / len(df)
                            print(f"       {obj_type}: {count:,} ({pct:.1f}%)")
                    
                    if self.verbose and chunk_num % max(1, n_chunks // 20) == 0:
                        pct = 100 * (line_idx + 1) / self.total_lines
                        print(f"     Processing chunk {chunk_num+1}/{n_chunks} ({pct:.1f}%)")
                    
                    yield df
                    chunk_num += 1
                    chunk_data = {col: [] for col in self.col_names}
                    chunk_rows = 0
            
            # Yield final partial chunk
            if chunk_rows > 0:
                df = pd.DataFrame(chunk_data)
                if 'type_code' in df.columns:
                    df['type'] = df['type_code'].apply(
                        lambda x: 'Galaxy' if str(x).strip().upper() == 'G' else 
                                 'Quasar' if str(x).strip().upper() == 'Q' else 
                                 'Unknown'
                    )
                yield df


# ========================================================================
# HDF5 CATALOG STREAMING READER - VERIFIED
# ========================================================================

class HDF5CatalogReader:
    """Memory-efficient HDF5 catalog reader."""
    
    def __init__(self, h5_file: str, chunk_size: int = 50000, verbose: bool = True):
        self.h5_file = h5_file
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        with h5py.File(h5_file, 'r') as h5f:
            self.total_rows = h5f.attrs.get('total_rows', 0)
            if self.total_rows == 0:
                first_key = list(h5f.keys())[0]
                self.total_rows = len(h5f[first_key])
            self.columns = list(h5f.keys())
        
        if verbose:
            print(f"[2/5] HDF5 Catalog Info:")
            print(f"     ✓ Total rows: {self.total_rows:,}")
            print(f"     ✓ Columns: {len(self.columns)}")
    
    def iter_chunks(self):
        n_chunks = int(np.ceil(self.total_rows / self.chunk_size))
        
        for chunk_num, start_idx in enumerate(range(0, self.total_rows, self.chunk_size)):
            end_idx = min(start_idx + self.chunk_size, self.total_rows)
            
            with h5py.File(self.h5_file, 'r') as h5f:
                chunk_dict = {}
                for col in self.columns:
                    chunk_dict[col] = h5f[col][start_idx:end_idx]
            
            df = pd.DataFrame(chunk_dict)
            
            if 'type_code' in df.columns:
                df['type'] = df['type_code'].apply(
                    lambda x: 'Galaxy' if str(x).strip().upper() == 'G' else 
                             'Quasar' if str(x).strip().upper() == 'Q' else 
                             'Unknown'
                )
            
            if self.verbose and chunk_num % max(1, n_chunks // 20) == 0:
                pct = 100 * end_idx / self.total_rows
                print(f"     Processing chunk {chunk_num+1}/{n_chunks} ({pct:.1f}%)")
            
            yield df


# ========================================================================
# 3D CROSS-MATCHING ENGINE
# ========================================================================

class StreamingCrossMatcher:
    """3D cross-matching that preserves type information."""
    
    def __init__(self, skymap: GWSkymapLoader, verbose: bool = True):
        self.skymap = skymap
        self.verbose = verbose
    
    def distance_cdf(self, d: float, distmu: float, distsigma: float,
                     distnorm: float, nsamp: int = 200) -> float:
        """Compute CDF P(D < d | pixel) via numerical integration."""
        d_grid = np.linspace(0, d, nsamp)
        pdf = distnorm * np.exp(-0.5 * ((d_grid - distmu) / distsigma)**2)
        cdf = np.trapz(pdf, d_grid)
        return cdf
    
    def crossmatch_chunk(self, df: pd.DataFrame,
                        prob: np.ndarray,
                        cred_levels: np.ndarray,
                        distmu: np.ndarray,
                        distsigma: np.ndarray,
                        distnorm: np.ndarray,
                        sky_credible: float = 0.9,
                        dist_credible: float = 0.9) -> Optional[pd.DataFrame]:
        """Cross-match chunk against skymap, preserving ALL columns including type."""
        
        if len(df) == 0:
            return None
        
        ra = df['RA'].values
        dec = df['Dec'].values
        dl = df['DL'].values
        
        try:
            ipix = self.skymap.get_healpix_indices(ra, dec)
        except Exception:
            return None
        
        # Sky credible region filter
        sky_inside = (cred_levels[ipix] <= sky_credible)
        
        # Distance filter
        dist_ok = np.zeros(len(df), dtype=bool)
        valid_dist = np.isfinite(dl) & (dl > 0)
        
        if np.sum(valid_dist) > 0 and self.skymap.is_3d:
            mu = distmu[ipix[valid_dist]]
            sig = distsigma[ipix[valid_dist]]
            norm = distnorm[ipix[valid_dist]]
            d = dl[valid_dist]
            
            cdf_vals = np.array([
                self.distance_cdf(d[i], mu[i], sig[i], norm[i])
                for i in range(len(d))
            ])
            
            lo_frac = (1.0 - dist_credible) / 2.0
            hi_frac = (1.0 + dist_credible) / 2.0
            dist_ok[valid_dist] = (cdf_vals >= lo_frac) & (cdf_vals <= hi_frac)
        else:
            dist_ok[valid_dist] = True
        
        # Combined 3D filter
        inside_3d = sky_inside & dist_ok
        
        if np.sum(inside_3d) == 0:
            return None
        
        # CRITICAL: .copy() preserves ALL columns including type_code and type
        matched = df[inside_3d].copy()
        matched['prob_sky'] = prob[ipix[inside_3d]]
        
        return matched


# ========================================================================
# MAIN WORKFLOW
# ========================================================================

def main_streaming_crossmatch(catalog_file: str,
                             skymap_fits: str,
                             output_fits: str = "matched_galaxies.fits",
                             skymap_credible: float = 0.9,
                             dist_credible: float = 0.9,
                             resume: bool = False,
                             verbose: bool = True) -> Tuple[int, str]:
    """Main workflow with verified galaxy/quasar classification."""
    
    print("\n" + "="*70)
    print("GLADE+ × GW Skymap Cross-Match - Galaxy & Quasar Classification")
    print("="*70 + "\n")
    
    start_time = time.time()
    checkpoint_mgr = CrossmatchCheckpoint(output_fits)
    
    # Load skymap
    skymap = GWSkymapLoader(skymap_fits, verbose=verbose)
    
    # Prepare skymap arrays
    if verbose:
        print(f"\n[3/5] Preparing skymap credible regions...")
    
    prob = skymap.get_prob_array()
    cred_levels = find_greedy_credible_levels(prob)
    
    if skymap.is_3d:
        distmu, distsigma, distnorm = skymap.get_distance_arrays()
    else:
        distmu = distsigma = distnorm = None
    
    if verbose:
        print(f"     ✓ Sky credible level: {skymap_credible*100:.0f}%")
        if skymap.is_3d:
            print(f"     ✓ Distance credible level: {dist_credible*100:.0f}%")
    
    # Set up catalog reader
    if verbose:
        print(f"\n[4/5] Detecting catalog format...")
    
    catalog_format = detect_catalog_format(catalog_file)
    
    if verbose:
        print(f"     ✓ Format: {'ASCII' if catalog_format == 'ascii' else 'HDF5'}")
    
    if catalog_format == 'ascii':
        reader = ASCIICatalogReader(catalog_file, verbose=verbose)
    else:
        reader = HDF5CatalogReader(catalog_file, verbose=verbose)
    
    # Stream and cross-match
    if verbose:
        print(f"\n[5/5] Cross-matching chunks...")
    
    matcher = StreamingCrossMatcher(skymap, verbose=verbose)
    matched_chunks = []
    n_total = 0
    n_matched = 0
    chunk_num = 0
    
    checkpoint = checkpoint_mgr.load() if resume else None
    start_chunk = checkpoint.get('chunk_num', 0) if checkpoint else 0
    
    try:
        for chunk_data in reader.iter_chunks():
            if chunk_num < start_chunk:
                chunk_num += 1
                continue
            
            matched_chunk = matcher.crossmatch_chunk(
                chunk_data,
                prob,
                cred_levels,
                distmu,
                distsigma,
                distnorm,
                sky_credible=skymap_credible,
                dist_credible=dist_credible
            )
            
            n_total += len(chunk_data)
            
            if matched_chunk is not None and len(matched_chunk) > 0:
                matched_chunks.append(matched_chunk)
                n_matched += len(matched_chunk)
            
            chunk_num += 1
            
            if chunk_num % 10 == 0:
                checkpoint_mgr.save({'chunk_num': chunk_num, 'n_matched': n_matched})
            
            if verbose and chunk_num % 50 == 0:
                elapsed = time.time() - start_time
                rate = chunk_num / elapsed
                print(f"     Chunk {chunk_num}: {n_matched:,} matched | Rate: {rate:.1f} chunks/sec")
    
    except KeyboardInterrupt:
        print(f"\n✗ Interrupted - checkpoint saved at chunk {chunk_num}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Combine and save results
    if verbose:
        print(f"\n[6/6] Writing outputs...")
    
    if len(matched_chunks) == 0:
        print(f"✗ No objects matched in credible region!")
        return 0, output_fits
    
    matched_df = pd.concat(matched_chunks, ignore_index=True)
    matched_df = matched_df.drop_duplicates().reset_index(drop=True)
    n_matched = len(matched_df)
    
    # VERIFY: Print type distribution
    if verbose and 'type' in matched_df.columns:
        print(f"\n     ✓ Object classification in matched results:")
        type_counts = matched_df['type'].value_counts()
        for obj_type, count in type_counts.items():
            pct = 100 * count / n_matched
            print(f"       {obj_type}: {count:,} ({pct:.1f}%)")
        print()
    
    # Save FITS
    matched_table = Table.from_pandas(matched_df)
    matched_table.write(output_fits, overwrite=True, format='fits')
    if verbose:
        print(f"     ✓ Wrote {n_matched:,} objects to {output_fits}")
    
    # Save HDF5
    output_h5 = output_fits.replace('.fits', '.h5')
    with pd.HDFStore(output_h5, 'w', complevel=5, complib='blosc') as store:
        store.put('galaxies', matched_df, format='table')
    if verbose:
        print(f"     ✓ Wrote {n_matched:,} objects to {output_h5}")
    
    # Save summary
    summary_file = output_fits.replace('.fits', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("GLADE+ × GW SKYMAP CROSS-MATCH SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total GLADE+ objects processed: {n_total:,}\n")
        f.write(f"Matched objects in credible region: {n_matched:,}\n")
        f.write(f"Match efficiency: {100*n_matched/n_total:.3f}%\n\n")
        f.write(f"Sky credible level: {skymap_credible*100:.0f}%\n")
        if skymap.is_3d:
            f.write(f"Distance credible level: {dist_credible*100:.0f}%\n")
        f.write(f"\nObject Classification:\n")
        if 'type' in matched_df.columns:
            type_counts = matched_df['type'].value_counts()
            for obj_type, count in type_counts.items():
                pct = 100 * count / n_matched
                f.write(f"  {obj_type}: {count:,} ({pct:.1f}%)\n")
        f.write(f"\nDistance Statistics:\n")
        f.write(f"  Range: {matched_df['DL'].min():.1f} - {matched_df['DL'].max():.1f} Mpc\n")
        f.write(f"\nSky Position:\n")
        f.write(f"  RA range: {matched_df['RA'].min():.1f} - {matched_df['RA'].max():.1f} deg\n")
        f.write(f"  Dec range: {matched_df['Dec'].min():.1f} - {matched_df['Dec'].max():.1f} deg\n")
    
    if verbose:
        print(f"     ✓ Wrote summary to {summary_file}")
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ SUCCESS!")
        print(f"{'='*70}")
        print(f"Matched objects: {n_matched:,}")
        print(f"Processing time: {elapsed/60:.1f} minutes")
        print(f"{'='*70}\n")
    
    checkpoint_mgr.delete()
    
    return n_matched, output_fits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLADE+ × GW cross-match with Galaxy/Quasar classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crossmatch.py GLADE_2.3.txt skymap.fits
  python crossmatch.py GLADE_2.3.txt skymap.fits -o output.fits --skymap-credible 0.9
        """
    )
    
    parser.add_argument("catalog", help="Path to GLADE+ catalog (.txt or .h5)")
    parser.add_argument("skymap", help="Path to GW skymap FITS file")
    parser.add_argument("-o", "--output", default="matched_galaxies.fits", 
                       help="Output filename (default: matched_galaxies.fits)")
    parser.add_argument("--skymap-credible", type=float, default=0.9,
                       help="Sky credible level (default: 0.9)")
    parser.add_argument("--dist-credible", type=float, default=0.9,
                       help="Distance credible level (default: 0.9)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    try:
        result = main_streaming_crossmatch(
            args.catalog,
            args.skymap,
            output_fits=args.output,
            skymap_credible=args.skymap_credible,
            dist_credible=args.dist_credible,
            resume=args.resume,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"✗ Cross-match failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
