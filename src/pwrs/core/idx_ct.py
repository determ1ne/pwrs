# column labels for changes table
CT_LABEL = 1  # change set label
CT_PROB = 2  # change set probability
CT_TABLE = 3  # type of table to be modified (see possible values below)
CT_ROW = 4  # number of the row to be modified (0 means all rows)
CT_COL = 5  # number of the column to be modified
# (for some values in CT_TABLE column, this can be a
# special code instead of an actual column index)
CT_CHGTYPE = 6  # type of parameter modification to be made
# (see possible values below)
CT_NEWVAL = 7  # quantity to use for replacement value, scale factor
# or shift amount

# named values for CT_TABLE entry
CT_TBUS = 1  # bus table
CT_TGEN = 2  # gen table
CT_TBRCH = 3  # branch table
CT_TAREABUS = 4  # area-wide change in bus table
CT_TAREAGEN = 5  # area-wide change in gen table
CT_TAREABRCH = 6  # area-wide change in branch table
CT_TLOAD = 7  # single bus load change
CT_TAREALOAD = 8  # area-wide bus load change
CT_TGENCOST = 9  # gencost table
CT_TAREAGENCOST = 10  # area-wide change in gencost table

# named values for CT_CHGTYPE entry
CT_REP = 1  # replace old value with new one in column CT_NEWVAL
CT_REL = 2  # multiply old value by factor in column CT_NEWVAL
CT_ADD = 3  # add value in column CT_NEWVAL to old value

# codes for CT_COL entry when CT_TABLE entry is CT_TLOAD or CT_TAREALOAD
CT_LOAD_ALL_PQ = 1  # all loads, real and reactive
CT_LOAD_FIX_PQ = 2  # only fixed loads, real and reactive
CT_LOAD_DIS_PQ = 3  # only dispatchable loads, real and reactive
CT_LOAD_ALL_P = 4  # all loads, real only
CT_LOAD_FIX_P = 5  # only fixed loads, real only
CT_LOAD_DIS_P = 6  # only dispatchable loads, real only

# codes for CT_COL entry when CT_TABLE entry is CT_TGENCOST or CT_TAREAGENCOST
CT_MODCOST_F = -1  # scale or shift cost function vertically
CT_MODCOST_X = -2  # scale or shift cost function horizontally


def idx_ct():
    return (
        CT_LABEL,
        CT_PROB,
        CT_TABLE,
        CT_TBUS,
        CT_TGEN,
        CT_TBRCH,
        CT_TAREABUS,
        CT_TAREAGEN,
        CT_TAREABRCH,
        CT_ROW,
        CT_COL,
        CT_CHGTYPE,
        CT_REP,
        CT_REL,
        CT_ADD,
        CT_NEWVAL,
        CT_TLOAD,
        CT_TAREALOAD,
        CT_LOAD_ALL_PQ,
        CT_LOAD_FIX_PQ,
        CT_LOAD_DIS_PQ,
        CT_LOAD_ALL_P,
        CT_LOAD_FIX_P,
        CT_LOAD_DIS_P,
        CT_TGENCOST,
        CT_TAREAGENCOST,
        CT_MODCOST_F,
        CT_MODCOST_X,
    )
