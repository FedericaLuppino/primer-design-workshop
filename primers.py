"""
primers.py — A primer design toolkit for the classroom.
========================================================

This module has two main functions:

  1. design_primers()
     Give it a chromosome and a region (start, end) and it will:
       - Fetch the DNA sequence from the Ensembl database (no download needed!)
       - Slide a window across that sequence to generate every possible primer
         candidate (called a "kmer") of the lengths you choose.
       - Optionally also generate primers from the reverse complement strand.

  2. analyze_primers()
     Give it the list of primers from design_primers() and it will:
       - Calculate the GC content of each primer.
       - Calculate the melting temperature (Tm) using the nearest-neighbour
         method (SantaLucia 1998) with a salt correction (Owczarzy 2008).
       - Keep only the primers that pass the quality thresholds you set.

Quick start
-----------
    from primers import design_primers, analyze_primers

    # 1. Generate primer candidates from a region on chromosome 2L
    candidates = design_primers(chrom="2L", start=5000, end=6000)

    # 2. Filter by GC content and melting temperature
    good_primers = analyze_primers(candidates)

    # 3. Look at the results
    for p in good_primers[:5]:
        print(p)

Requirements
------------
    pip install biopython requests
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List
from Bio.SeqUtils import MeltingTemp as mt
from Bio import SeqIO


# =============================================================================
# The Primer data structure
# =============================================================================

@dataclass
class Primer:
    """
    Stores all the information about a single primer candidate.

    You can access each piece of information using a dot, for example:
        primer.sequence      # the DNA sequence
        primer.tm            # melting temperature in °C
        primer.chrom         # chromosome name

    Attributes
    ----------
    sequence : str
        The DNA sequence of the primer (e.g. "ATCGATCGATCGATCGATCG").
    length : int
        How long the primer is in base pairs.
    chrom : str
        Which chromosome the primer comes from (e.g. "2L", "X").
    start : int
        Start position on the chromosome (1-based, inclusive).
    end : int
        End position on the chromosome (1-based, inclusive).
    strand : str
        "+" means forward strand, "-" means reverse complement strand.
    genome : str
        The species / genome version this primer was designed from.
    gc_content : float or None
        GC content as a percentage, e.g. 52.0 means 52%.
        This is None until you run analyze_primers().
    tm : float or None
        Melting temperature in degrees Celsius.
        This is None until you run analyze_primers().
    """
    sequence:    str
    length:      int
    chrom:       str
    start:       int
    end:         int
    strand:      str
    genome:      str
    gc_content:  Optional[float] = field(default=None, repr=True)
    tm:          Optional[float] = field(default=None, repr=True)


# =============================================================================
# Internal helper functions (students don't need to call these directly)
# =============================================================================

def _reverse_complement(sequence: str) -> str:
    """
    Return the reverse complement of a DNA sequence.

    Example:
        _reverse_complement("ATCG")  →  "CGAT"

    In PCR, primers on the reverse strand are always written as their
    reverse complement so that they are read 5' → 3'.
    """
    # Map each base to its complement
    complement_table = str.maketrans("ACGTacgt", "TGCAtgca")
    complemented = sequence.translate(complement_table)
    # Reverse it to get 5' → 3' direction
    return complemented[::-1]


def _fetch_sequence_from_ensembl(chrom: str, start: int, end: int,
                                  genome_path: str) -> str:
    """
    Gets the requested coordinates from the existing, downloaded file. 
    This is suffcient for Drosophila, where genomes are small and allows
    us to manually cross-check our sequences against the file

    Parameters
    ----------
    chrom : str
        Chromosome name, e.g. "2L" for Drosophila or "chr1" for human.
    start : int
        Start position (1-based, inclusive).
    end : int
        End position (1-based, inclusive).
    genome_path : str
        Path to the genome to use for reference

    Returns
    -------
    str
        The DNA sequence in uppercase letters (A, T, C, G, or N for unknown).

    Raises
    ------
    ValueError
        If Ensembl cannot find the region or the request fails.
    """
    sequence = ''
    chr_sizes = {}
    try:
        for chromosome in SeqIO.parse(genome_path,'fasta'):
            chr_sizes[chromosome.id] = len(chromosome.seq)
            if chromosome.id != chrom:
                continue
            sequence = str(chromosome.seq[start:end])
    except Exception as e:
        print(f'Encountered error during sequence extraction: {e}')
    if not sequence:
        print(f'Did not find any hits for {chrom}:{start}-{end}.\nEncountered chromosomes:')
        for chr_id, size in chr_sizes:
            print(f'\t{chr_id}:\t{size}')
    return sequence


# =============================================================================
# Function 1: Generate primer candidates from a genomic region
# =============================================================================

def design_primers(
    chrom: str,
    start: int,
    end: int,
    genome_path: str = 'd_melanogaster.filtered.fasta',
    kmer_lengths: Optional[List[int]] = None,
    include_reverse_complement: bool = True,
) -> List[Primer]:
    """
    Extract a genomic sequence and generate all possible primer candidates.

    This function contacts the Ensembl database to retrieve the DNA sequence
    at the region you specify, then slides a window of different sizes across
    it to produce every possible primer candidate (a "kmer").

    Parameters
    ----------
    chrom : str
        Chromosome name, e.g. "2L" for Drosophila, "X", "3R", etc.
    start : int
        Start position on the chromosome (1-based, inclusive).
    end : int
        End position on the chromosome (1-based, inclusive).
    genome_path : str, optional
        Path to the genome to use for reference
    kmer_lengths : list of int, optional
        Which primer lengths (in bp) to generate.
        Default is [20, 21, 22, 23].
        You can try longer primers, e.g. [25, 26, 27], but they are
        typically harder to work with in the lab.
    include_reverse_complement : bool, optional
        If True (the default), also generate primers from the reverse
        complement strand. In PCR you need primers on both strands!

    Returns
    -------
    list of Primer
        A list of Primer objects. Each one has:
            .sequence  — the DNA sequence
            .length    — how long it is
            .chrom     — chromosome
            .start     — genomic start (1-based)
            .end       — genomic end (1-based)
            .strand    — "+" (forward) or "-" (reverse complement)
            .genome    — which genome was used
        The .gc_content and .tm fields will be None — fill them in
        by passing this list to analyze_primers().

    Examples
    --------
    >>> # Design primers in a 1 kb region on chromosome 2L
    >>> candidates = design_primers(chrom="2L", start=5000, end=6000)
    >>> print(f"Total candidates: {len(candidates)}")

    >>> # Only generate 20-mer primers, forward strand only
    >>> candidates = design_primers(
    ...     chrom="2L",
    ...     start=5000,
    ...     end=6000,
    ...     kmer_lengths=[20],
    ...     include_reverse_complement=False,
    ... )
    """
    # ---- Set default kmer lengths ----
    if kmer_lengths is None:
        kmer_lengths = [20, 21, 22, 23]

    # ---- Validate inputs ----
    if start >= end:
        raise ValueError(
            f"'start' ({start}) must be smaller than 'end' ({end})."
        )
    if any(k < 10 for k in kmer_lengths):
        raise ValueError(
            "Kmer lengths below 10 bp are too short to be useful as primers."
        )

    # ---- Fetch the sequence from Ensembl ----
    print(f"Fetching {chrom}:{start}-{end} from reference genome ({genome_path})...")
    sequence = _fetch_sequence_from_ensembl(chrom, start, end, genome_path)
    region_length = len(sequence)
    print(f"  Got {region_length} bp.")

    # ---- Slide a window across the sequence for each kmer length ----
    primers = []

    for kmer_len in kmer_lengths:
        if kmer_len > region_length:
            print(
                f"  Warning: kmer length {kmer_len} bp is longer than the region "
                f"({region_length} bp). Skipping this length."
            )
            continue

        n_positions = region_length - kmer_len + 1

        for i in range(n_positions):
            kmer_seq = sequence[i : i + kmer_len]

            # Skip positions with ambiguous bases (N = unknown nucleotide)
            if "N" in kmer_seq:
                continue

            # Convert window position to genomic coordinates (1-based)
            kmer_start = start + i
            kmer_end   = start + i + kmer_len - 1

            # Forward strand primer
            primers.append(Primer(
                sequence = kmer_seq,
                length   = kmer_len,
                chrom    = chrom,
                start    = kmer_start,
                end      = kmer_end,
                strand   = "+",
                genome   = genome_path,
            ))

            # Reverse complement primer (same genomic location, opposite strand)
            if include_reverse_complement:
                primers.append(Primer(
                    sequence = _reverse_complement(kmer_seq),
                    length   = kmer_len,
                    chrom    = chrom,
                    start    = kmer_start,
                    end      = kmer_end,
                    strand   = "-",
                    genome   = genome_path,
                ))

    strands_msg = "forward + reverse complement" if include_reverse_complement else "forward only"
    print(
        f"  Generated {len(primers)} primer candidates "
        f"({len(kmer_lengths)} length(s): {kmer_lengths}, {strands_msg})."
    )
    return primers


# =============================================================================
# Function 2: Calculate GC content and Tm, then filter
# =============================================================================

def analyze_primers(
    primers: List[Primer],
    gc_min:  float = 40.0,
    gc_max:  float = 60.0,
    tm_min:  float = 51.0,
    tm_max:  float = 56.5,
    Na:      float = 50.0,
    Mg:      float = 2.0,
    K:       float = 0.0,
    Tris:    float = 0.0,
    dNTPs:   float = 0.2,
    dnac1:   float = 0.016,
) -> List[Primer]:
    """
    Calculate GC content and melting temperature (Tm) for each primer,
    and return only the ones that pass the quality thresholds.

    What is GC content?
    -------------------
    The fraction of bases that are G or C, expressed as a percentage.
    Primers with 40–60% GC tend to bind reliably without being too
    "sticky". Very GC-rich primers can form unwanted secondary structures.

    What is Tm?
    -----------
    The melting temperature is the temperature at which half of the
    primer-DNA pairs have come apart. A good PCR primer typically has
    a Tm between 50°C and 65°C. Pairs of primers should have similar
    Tm values so they work at the same annealing temperature.

    How is Tm calculated here?
    --------------------------
    We use the "nearest-neighbour" method (SantaLucia 1998), which
    considers not just how many G/C and A/T bases there are, but also
    which bases sit next to each other (because neighbouring bases
    affect how tightly they stack). The salt correction (Owczarzy 2008)
    accounts for the concentrations of ions in the PCR reaction, which
    also affect how tightly the primer binds.

    Parameters
    ----------
    primers : list of Primer
        The output of design_primers().
    gc_min : float, optional
        Minimum allowed GC content (%). Default: 40.0.
    gc_max : float, optional
        Maximum allowed GC content (%). Default: 60.0.
    tm_min : float, optional
        Minimum allowed melting temperature in °C. Default: 51.0.
    tm_max : float, optional
        Maximum allowed melting temperature in °C. Default: 56.5.
    Na : float, optional
        Sodium concentration in mM. Default: 50 mM.
    Mg : float, optional
        Magnesium concentration in mM. Default: 2 mM.
    K : float, optional
        Potassium concentration in mM. Default: 0 mM.
    Tris : float, optional
        Tris buffer concentration in mM. Default: 0 mM.
    dNTPs : float, optional
        dNTP concentration in mM. Default: 0.2 mM.
    dnac1 : float, optional
        Primer (oligonucleotide) concentration in nM. Default: 0.016 nM.

    Returns
    -------
    list of Primer
        A new list containing only the primers that pass all filters.
        Each Primer now has .gc_content and .tm filled in.
        The original list passed in is not modified.

    Examples
    --------
    >>> candidates = design_primers("2L", 5000, 6000)
    >>> good = analyze_primers(candidates)
    >>> print(f"{len(good)} primers passed the filters")

    >>> # Use stricter GC filter and wider Tm window
    >>> good = analyze_primers(candidates, gc_min=45, gc_max=55, tm_min=50, tm_max=58)

    >>> # Access individual primer properties
    >>> for p in good[:3]:
    ...     print(f"{p.sequence}  GC={p.gc_content}%  Tm={p.tm}°C  [{p.chrom}:{p.start}-{p.end} {p.strand}]")
    """
    if not primers:
        print("Warning: the input list of primers is empty.")
        return []

    passing = []
    n_gc_fail = 0
    n_tm_fail = 0
    n_error   = 0

    for primer in primers:
        # ---- GC content ----
        # Count G and C bases, divide by total length, convert to %
        n_gc = sum(1 for base in primer.sequence if base in "GC")
        gc = round(n_gc / primer.length * 100, 1)

        if not (gc_min <= gc <= gc_max):
            n_gc_fail += 1
            continue

        # ---- Melting temperature (nearest-neighbour thermodynamics) ----
        try:
            tm_value = mt.Tm_NN(
                primer.sequence,
                nn_table = mt.DNA_NN3,  # SantaLucia 1998 parameters
                Na       = Na,
                K        = K,
                Tris     = Tris,
                Mg       = Mg,
                dNTPs    = dNTPs,
                dnac1    = dnac1,
                dnac2    = 0,           # single-stranded (no complementary strand)
                saltcorr = 6,           # Owczarzy 2008 salt correction
            )
            tm = round(tm_value, 1)
        except Exception:
            # A very unusual sequence (e.g. all the same base) can cause
            # the Tm calculation to fail; we skip those.
            n_error += 1
            continue

        if not (tm_min <= tm <= tm_max):
            n_tm_fail += 1
            continue

        # ---- This primer passed all filters: return a copy with GC and Tm filled in ----
        passing.append(dataclasses.replace(primer, gc_content=gc, tm=tm))

    print(
        f"\nFilter results ({len(primers)} primers in → {len(passing)} passed):\n"
        f"  Failed GC filter  (not {gc_min}–{gc_max}%)  : {n_gc_fail}\n"
        f"  Failed Tm filter  (not {tm_min}–{tm_max}°C) : {n_tm_fail}\n"
        f"  Tm calculation errors                        : {n_error}"
    )
    return passing
