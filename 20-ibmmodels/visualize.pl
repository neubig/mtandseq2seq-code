#!/usr/bin/perl

use utf8;
use List::Util qw( max min sum );
use strict;
binmode STDOUT, ":utf8";

if(@ARGV < 3 or @ARGV > 5) {
    print "Usage: visualize.pl FFILE EFILE AFILE [ADDPOS] [REVERSE]\n";
    exit 1;
}

open EFILE, "<:utf8", $ARGV[0] or die "$ARGV[0]: $!";
open FFILE, "<:utf8", $ARGV[1] or die "$ARGV[1]: $!";
open AFILE, "<:utf8", $ARGV[2] or die "$ARGV[2]: $!";

sub asciilength {
    my $str = shift;
    my $ret = 0;
    for(split(//, $str)) {
        if(/(\p{InHiragana}|\p{InKatakana}|\p{InCJKUnifiedIdeographs}|[ａ-ｚＡ-Ｚ０-９！”＃＄％＆’（）「」『』、。々])/) {
            $ret += 2;
        } else {
            $ret += 1;
        }
    }
    return $ret;
}

my ($i, $j);
my (@actives, $actmax, $estr, $fstr, $astr);
while($estr = <EFILE> and $fstr = <FFILE> and $astr = <AFILE>) {
    chomp $estr; chomp $fstr; chomp $astr;
    # if(not $astr) { next; print "\n"; }
    my %active = map { my ($e,$f) = split(/-/); my $id = "".$e."-".$f; $id => 1 } split(/ /,$astr);
    my @e = split(/ /,$estr);
    my $elen = max( map { asciilength($_) } @e );
    my @f = split(/ /,$fstr);
    my $flen = max( map { length($_) } @f );
    for(0 .. $flen) {
        print " " for(0 .. $elen);
        my $pos = $flen-$_;
        for(@f) {
            if($pos<length($_)) {
                my $str = substr($_,length($_)-$pos-1,1);
                if(asciilength($str) == 1) { print " "; }
                print $str;
            } else {
				print "  ";
            }
        }
        print "\n";
    }
    foreach my $i (0 .. $#e) {
        print " " for(1 .. ($elen-asciilength($e[$i])));
        print "$e[$i] ";
        foreach my $j (0 .. $#f) {
            my $id = "$i-$j";
            print ($active{$id}?" X":" .");
        }
        print "\n";
    }
    print "\n";
}
