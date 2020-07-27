//
// Created by kihiro on 7/27/20.
//

#include "legion.h"
#include "H5Cpp.h"
#include "my_mapper.h"

using namespace std;
using namespace Legion;
using namespace H5;

enum TaskIDs {
    TID_TOP_LEVEL,
    TID_WRITE,
};

enum FieldIDs {
    FID_DUMMY,
};

constexpr unsigned N = 50;
constexpr unsigned Npart = 2;
constexpr char FILENAME[] = "out.h5";

typedef Legion::FieldAccessor< READ_ONLY, double, 1, Legion::coord_t,
    Realm::AffineAccessor<double, 1, Legion::coord_t> > AffAccROrtype;
typedef Legion::FieldAccessor< READ_WRITE, double, 1, Legion::coord_t,
    Realm::AffineAccessor<double, 1, Legion::coord_t> > AffAccRWrtype;
typedef Legion::FieldAccessor< WRITE_DISCARD, double, 1, Legion::coord_t,
    Realm::AffineAccessor<double, 1, Legion::coord_t> > AffAccWDrtype;

void write_task(const Task *task, const vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {

    assert(regions.size()==2);

    AffAccROrtype acc_src(regions[0], FID_DUMMY, sizeof(double));
    AffAccWDrtype acc_dest(regions[1], FID_DUMMY, sizeof(double));
    Domain domain = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
    for (Domain::DomainPointIterator itr(domain); itr; itr++) {
        acc_dest[*itr] = acc_src[*itr];
    }
}

void top_level_task(const Task *task, const vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime) {

    const Rect<1> rect(0, N-1);
    IndexSpace is = runtime->create_index_space(ctx, rect);
    runtime->attach_name(is, "index_space");

    FieldSpace fs = runtime->create_field_space(ctx);
    runtime->attach_name(fs, "field_space");

    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double), FID_DUMMY);
    runtime->attach_name(fs, FID_DUMMY, "dummy_field");

    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(lr, "logical_region");

    double value = 1;
    runtime->fill_field(ctx, lr, lr, FID_DUMMY, &value, sizeof(double));

    IndexSpace color_space = runtime->create_index_space(ctx, Rect<1>(0, Npart-1));
    IndexPartition ip = runtime->create_equal_partition(ctx, lr.get_index_space(), color_space);
    runtime->attach_name(ip, "index_partition");
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    runtime->attach_name(lp, "logical_partition");

    LogicalRegion file_lr = runtime->create_logical_region(ctx,
        lr.get_index_space(), lr.get_field_space());
    runtime->attach_name(file_lr, "file_logical_region");
    LogicalPartition file_lp = runtime->get_logical_partition(ctx, file_lr, ip);
    runtime->attach_name(file_lp, "file_logical_partition");

    map<FieldID, const char*> field_map {
        {FID_DUMMY, "Dummy"}
    };
    AttachLauncher attach_launcher(EXTERNAL_HDF5_FILE, file_lr, file_lr, false, true);
    attach_launcher.attach_hdf5(FILENAME, field_map, LEGION_FILE_READ_WRITE);
    PhysicalRegion pr = runtime->attach_external_resource(ctx, attach_launcher);

    IndexTaskLauncher launcher(TID_WRITE, color_space, TaskArgument(), ArgumentMap());
    {
        RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, lr);
        req.add_field(FID_DUMMY);
        launcher.add_region_requirement(req);
    }
    {
        RegionRequirement req(file_lp, 0, WRITE_DISCARD, EXCLUSIVE, file_lr);
        req.add_field(FID_DUMMY);
        launcher.add_region_requirement(req);
    }
    launcher.tag = MT_HDF_OUTPUT;
    runtime->execute_index_space(ctx, launcher);
}

int main(int argc, char *argv[]) {

    Runtime::set_top_level_task_id(TID_TOP_LEVEL);
    {
        TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_replicable(true);
        Runtime::preregister_task_variant<top_level_task> (registrar, "top_level_task");
    }

    {
        TaskVariantRegistrar registrar(TID_WRITE, "write_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<write_task> (registrar, "write_task");
    }

    H5File file(FILENAME, H5F_ACC_TRUNC);
    hsize_t dim_dset[1];
    hsize_t rank = 1;
    dim_dset[0] = N;
    DataSpace dspace(rank, dim_dset);
    file.createDataSet("Dummy", PredType::NATIVE_DOUBLE, dspace);
    file.close();

    register_mappers();
    return Runtime::start(argc, argv);
}
