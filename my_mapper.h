//
// Created by kihiro on 2/19/20.
//

#ifndef DG_MY_MAPPERS_H
#define DG_MY_MAPPERS_H

enum MappingTags {
    // the default mapper has tags up to (1<<4)
    MT_RANK_DISPATCH = (1<<5),
    MT_HDF_OUTPUT = (1<<6),
    MT_AOS = (1<<7), // TODO remove this tag since this is a hard constraint that should be set at registration
};

void register_mappers();

#endif //DG_MY_MAPPERS_H
