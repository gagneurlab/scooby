from intervaltree import IntervalTree
import numpy as np
import gzip


class Transcriptome:
    def __init__(self, gtf_file, use_geneid=False):
        self.genes = {}
        self.use_geneid = use_geneid
        self.read_gtf(gtf_file)

    def read_gtf(self, gtf_file):
        if gtf_file[-3:] == ".gz":
            gtf_in = gzip.open(gtf_file, "rt")
        else:
            gtf_in = open(gtf_file)

        # ignore header
        line = gtf_in.readline()
        while line[0] == "#":
            line = gtf_in.readline()

        while line:
            a = line.split("\t")
            if a[2] == "exon":
                chrom = a[0]
                start = int(a[3])
                end = int(a[4])
                strand = a[6]
                kv = gtf_kv(a[8])
                gene_id = kv["gene_name"] if not self.use_geneid else kv["gene_id"]

                # initialize gene
                if gene_id not in self.genes:
                    self.genes[gene_id] = Gene(chrom, strand, kv)

                # add exon
                self.genes[gene_id].add_exon(start - 1, end)

            line = gtf_in.readline()

        gtf_in.close()

    def bedtool_exon(self):
        # assemble sequence bedtool
        bed_lines = []
        for gene_id, gene in self.genes.items():
            for exon in gene.get_exons():
                exon_line = "%s %d %d %s . %s" % (
                    gene.chrom,
                    exon.begin,
                    exon.end,
                    gene_id,
                    gene.strand,
                )
                bed_lines.append(exon_line)
        genes_bedt = pybedtools.BedTool("\n".join(bed_lines), from_string=True)
        return genes_bedt

    def bedtool_span(self):
        # assemble sequence bedtool
        bed_lines = []
        for gene_id, gene in self.genes.items():
            gene_start, gene_end = gene.span()
            span_line = "%s %d %d %s . %s" % (
                gene.chrom,
                gene_start,
                gene_end,
                gene_id,
                gene.strand,
            )
            bed_lines.append(span_line)
        genes_bedt = pybedtools.BedTool("\n".join(bed_lines), from_string=True)
        return genes_bedt

    def write_bed_exon(self, bed_file):
        pass

    def write_bed_span(self, bed_file):
        pass


################################################################################
# Methods
################################################################################
def gtf_kv(s):
    """Convert the last gtf section of key/value pairs into a dict."""
    d = {}

    a = s.split(";")
    for key_val in a:
        if key_val.strip():
            eq_i = key_val.find("=")
            if eq_i != -1 and key_val[eq_i - 1] != '"':
                kvs = key_val.split("=")
            else:
                kvs = key_val.split()

            key = kvs[0]
            if kvs[1][0] == '"' and kvs[-1][-1] == '"':
                val = (" ".join(kvs[1:]))[1:-1].strip()
            else:
                val = (" ".join(kvs[1:])).strip()

            d[key] = val

    return d


class Gene:
    """Class for managing genes in an isoform-agnostic way, taking
    the union of exons across isoforms."""

    def __init__(self, chrom, strand, kv):
        self.chrom = chrom
        self.strand = strand
        self.kv = kv
        self.exons = IntervalTree()

    def add_exon(self, start, end):
        """BED 0-indexing assumed."""
        self.exons[start:end] = True

    def get_exons(self):
        self.exons.merge_overlaps()
        return sorted(self.exons)

    def midpoint(self):
        positions = []
        for exon in self.get_exons():
            positions += range(exon.begin, exon.end)
        midp = int(np.mean(positions))
        return midp

    def span(self):
        exon_starts = [exon.begin for exon in self.exons]
        exon_ends = [exon.end for exon in self.exons]
        return min(exon_starts), max(exon_ends)

    def output_slice(self, seq_start, seq_len, model_stride, span=False, sliced=True):
        gene_slice = []

        if span:
            gene_start, gene_end = self.span()

            # clip left boundaries
            gene_seq_start = max(0, gene_start - seq_start)
            gene_seq_end = max(0, gene_end - seq_start)

            # requires >50% overlap
            slice_start = int(np.round(gene_seq_start / model_stride))
            slice_end = int(np.round(gene_seq_end / model_stride))

            # clip right boundaries
            slice_max = int(seq_len / model_stride)
            slice_start = min(slice_start, slice_max)
            slice_end = min(slice_end, slice_max)

            if sliced:
                gene_slice = range(slice_start, slice_end)
            else:
                gene_slice = [(slice_start, slice_end)]

        else:
            for exon in self.get_exons():
                # clip left boundaries
                exon_seq_start = max(0, exon.begin - seq_start)
                exon_seq_end = max(0, exon.end - seq_start)

                # requires >50% overlap
                slice_start = int(np.round(exon_seq_start / model_stride))
                slice_end = int(np.round(exon_seq_end / model_stride))

                # clip right boundaries
                slice_max = int(seq_len / model_stride)
                slice_start = min(slice_start, slice_max)
                slice_end = min(slice_end, slice_max)

                if sliced:
                    gene_slice.extend(range(slice_start, slice_end))
                else:
                    gene_slice.append((slice_start, slice_end))

        return np.array(gene_slice) if sliced else gene_slice
