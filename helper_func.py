def an_df_filler(in_f, row_num, in_dat, col_name):
    for count, a in enumerate(in_dat):
        if len(col_name) > 1:
            in_f.loc[row_num, col_name[count] + str(count + 1)] = a
        else:
            in_f.loc[row_num, col_name[0] + str(count + 1)] = a

    return in_f


