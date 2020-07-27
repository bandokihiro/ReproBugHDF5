//
// Created by kihiro on 5/9/20.
//

#include <iostream>
#include <map>

#include "legion.h"

#include "mappers/default_mapper.h"
#include "my_mapper.h"

using namespace std;
using namespace Legion;
using namespace Legion::Mapping;

/**************************************************************************************************/
/* MY MAPPER INTERFACE                                                                            */
/**************************************************************************************************/

static Logger log_mapper("my_mapper");

enum {
    SFID_NAIVE = 1000,
    SFID_RANK_DISPATCH,
    SFID_UNIQUE,
};

class ShardingNaive : public ShardingFunctor {
  public:
    ShardingNaive(void);
    ShardingNaive(const ShardingNaive &rhs);
    virtual ~ShardingNaive(void);
    ShardingNaive &operator=(const ShardingNaive &rhs);
    virtual ShardID shard(const DomainPoint &point,
        const Domain &full_space, const size_t total_shards);
};

class ShardingRankDispatch : public ShardingFunctor {
  public:
    ShardingRankDispatch(void);
    ShardingRankDispatch(const ShardingRankDispatch &rhs);
    virtual ~ShardingRankDispatch(void);
    ShardingRankDispatch &operator=(const ShardingRankDispatch &rhs);
    virtual ShardID shard(const DomainPoint &point,
        const Domain &full_space, const size_t total_shards);
};

class ShardingUnique : public ShardingFunctor {
  public:
    ShardingUnique(void);
    ShardingUnique(const ShardingUnique &rhs);
    virtual ~ShardingUnique(void);
    ShardingUnique &operator=(const ShardingUnique &rhs);
    virtual ShardID shard(const DomainPoint &point,
        const Domain &full_space, const size_t total_shards);
};

class MyMapper : public Mapping::DefaultMapper {
  public:
    MyMapper(Runtime *runtime, Machine machine, Processor local,
        const char *mapper_name);

    void default_policy_select_instance_fields(MapperContext ctx,  const RegionRequirement &req,
        const set<FieldID> &needed_fields, vector<FieldID> &fields) override;

    void slice_task(const MapperContext ctx, const Task &task, const SliceTaskInput &input,
        SliceTaskOutput &output) override;

    void map_task(const MapperContext ctx, const Task &task, const MapTaskInput &input,
        MapTaskOutput &output) override;

#ifdef USE_DCR
    void select_sharding_functor(const MapperContext ctx, const Task &task,
        const SelectShardingFunctorInput &input, SelectShardingFunctorOutput& output) override;
#endif

  protected:
    void default_policy_select_target_processors(Mapping::MapperContext ctx,
        const Task &task, vector<Processor> &target_procs) override;

    void map_task_colocation(const MapperContext ctx, const Task &task,
        const ExecutionConstraintSet &constraints,
        MapTaskOutput &output, vector<bool> &done_regions);
};

/**************************************************************************************************/
/* MY SHARDING FUNCTORS IMPLEMENTATION                                                            */
/**************************************************************************************************/

ShardingNaive::ShardingNaive(void) : ShardingFunctor() {}

ShardingNaive::ShardingNaive(const ShardingNaive &rhs) : ShardingFunctor() {
    // should never be called
    assert(false);
}

ShardingNaive::~ShardingNaive(void) {}

ShardingNaive &ShardingNaive::operator=(const ShardingNaive &rhs) {
    // should never be called
    assert(false);
    return *this;
}

ShardID ShardingNaive::shard(const DomainPoint &point, const Domain &full_space,
      const size_t total_shards) {
    // brutally assert
    assert(point.get_dim()==1);
    // for single point task launcher, assign to the same shard for now
    if (full_space.get_volume()==1) {
      return 0;
    }
    /* This is a simple heuristic that works well when the number of sub-regions is a multiple of
     * the number of nodes, which with this mapper is the number of shards.
     */
    return (unsigned) (point[0] / (unsigned) (full_space.get_volume() / total_shards))
        % (unsigned) total_shards;
}

//--------------------------------------------------------------------------------------------------

ShardingRankDispatch::ShardingRankDispatch(void) : ShardingFunctor() {}

ShardingRankDispatch::ShardingRankDispatch(const ShardingRankDispatch &rhs) : ShardingFunctor() {
    // should never be called
    assert(false);
}

ShardingRankDispatch::~ShardingRankDispatch(void) {}

ShardingRankDispatch &ShardingRankDispatch::operator=(const ShardingRankDispatch &rhs) {
    // should never be called
    assert(false);
    return *this;
}

ShardID ShardingRankDispatch::shard(const DomainPoint &point, const Domain &full_space,
        const size_t total_shards) {
    assert(point.get_dim()==1); // the dimension of the launch space should be one
    assert(full_space.get_volume()>1); // at least 2 lvl2 launches
    return (unsigned) point[0] % (unsigned) total_shards; // dispatch one task to each shard, ie a node-level launch
}

//--------------------------------------------------------------------------------------------------

ShardingUnique::ShardingUnique(void) : ShardingFunctor() {}

ShardingUnique::ShardingUnique(const ShardingUnique &rhs) : ShardingFunctor() {
    // should never be called
    assert(false);
}

ShardingUnique::~ShardingUnique(void) {}

ShardingUnique &ShardingUnique::operator=(const ShardingUnique &rhs) {
    // should never be called
    assert(false);
    return *this;
}

ShardID ShardingUnique::shard(const DomainPoint &point, const Domain &full_space,
        const size_t total_shards) {
    return 0;
}

/**************************************************************************************************/
/* MY MAPPER IMPLEMENTATION                                                                       */
/**************************************************************************************************/

MyMapper::MyMapper(Runtime *runtime, Machine machine, Processor local, const char *mapper_name)
        : DefaultMapper(runtime->get_mapper_runtime(), machine, local, mapper_name) {}

void MyMapper::default_policy_select_instance_fields(MapperContext ctx,
        const RegionRequirement &req, const set<FieldID> &needed_fields, vector<FieldID> &fields) {
    /* This function is called in default_create_custom_instances.
     * The default mapper add the entire field space of req.region when total_nodes=1.
     * I don't want this inconsistent behavior between multiple nodes and single node.
     * Only add the needed_fields to fields.
     */
    fields.insert(fields.end(), needed_fields.begin(), needed_fields.end());
}

void MyMapper::slice_task(const MapperContext ctx, const Task &task,
        const SliceTaskInput &input, SliceTaskOutput &output) {
#ifdef USE_DCR
    // log_mapper.print("Task %s", task.get_task_name());
    // log_mapper.print("Input domain %s", to_string(input.domain).c_str());

    vector<VariantID> variants;
    runtime->find_valid_variants(ctx, task.task_id, variants);

    // the default mapper does something here in case the variant can use PROC_SET
    // I don't care about this for now

    // the default mapper check if the task is a must epoch task before switching on the kind
    // I also don't care about that, just brutally assert
    assert(task.target_proc.kind() == Processor::LOC_PROC ||
           task.target_proc.kind() == Processor::OMP_PROC);

    // some aliases because the following code is mostly pasted from the default mapper
    map<Domain, vector<TaskSlice>> &cached_slices = cpu_slices_cache;
    vector<Processor> &local = local_cpus;
    if (task.target_proc.kind()==Processor::OMP_PROC) {
        cached_slices = omp_slices_cache;
        local = local_omps;
    }

    // check the cache
    map<Domain, vector<TaskSlice> >::const_iterator finder = cached_slices.find(input.domain);
    if (finder != cached_slices.end()) {
        output.slices = finder->second;
        return;
    }

    Machine::ProcessorQuery all_procs(machine);
    all_procs.only_kind(local[0].kind());
    // in DCR, we want only the local processors to be assigned to a shard
    all_procs.local_address_space();
    vector<Processor> procs(all_procs.begin(), all_procs.end());

    // my input domain is 1D for now
    assert(input.domain.get_dim() == 1);
    DomainT<1, coord_t> point_space = input.domain;
    Point<1, coord_t> num_blocks = default_select_num_blocks<1>(procs.size(), point_space.bounds);
    default_decompose_points<1>(point_space, procs, num_blocks,
        false/*recurse*/, stealing_enabled, output.slices);
#else
    DefaultMapper::slice_task(ctx, task, input, output);
#endif
}

void MyMapper::map_task_colocation(const MapperContext ctx, const Task &task,
        const ExecutionConstraintSet &constraints,
        MapTaskOutput &output, vector<bool> &done_regions) {
    const vector<ColocationConstraint> &coloc_constraints = constraints.colocation_constraints;
    const TaskLayoutConstraintSet &task_lay_cons =
        runtime->find_task_layout_constraints(ctx, task.task_id, output.chosen_variant);

    for (unsigned icol=0; icol<coloc_constraints.size(); icol++) {
        // log_mapper.print("Creating instance %d with colocation constraints in task %s",
        //     icol, task.get_task_name());

        const ColocationConstraint &constraint = coloc_constraints[icol];

        vector<FieldID> fields;
        for (set<FieldID>::const_iterator it=constraint.fields.begin();
             it!=constraint.fields.end(); it++) {
            fields.push_back(*it);
        }

        vector<LogicalRegion> target_regions;
        vector<unsigned> region_indexes;
        for (set<unsigned>::const_iterator it=constraint.indexes.begin();
             it!=constraint.indexes.end(); it++) {
            region_indexes.push_back(*it);
            target_regions.push_back(task.regions[*it].region);
        }
        /* Since this is a colocation constraint, I assume all region requirements
         * invovled shared the same set of constraints. So I use the first the
         * first one as reference.
         */
        const unsigned ref_idx = region_indexes[0];

        Memory target_memory = default_policy_select_target_memory(ctx,
            task.target_proc, task.regions[ref_idx]);

        LayoutConstraintSet layout_constraints;
        unsigned count_layout_constraints = 0;
        for (multimap<unsigned, LayoutConstraintID>::const_iterator
            lay_it=task_lay_cons.layouts.lower_bound(ref_idx);
            lay_it!=task_lay_cons.layouts.upper_bound(ref_idx);
            lay_it++) {
            layout_constraints = runtime->find_layout_constraints(ctx, lay_it->second);
            count_layout_constraints++;
        }
        assert(count_layout_constraints<=1);
        layout_constraints
            .add_constraint(FieldConstraint(fields,
                true /*contiguous fields*/, true /*fields in order*/))
            .add_constraint(MemoryConstraint(target_memory.kind()));

        PhysicalInstance result;
        bool created;
        if (!runtime->find_or_create_physical_instance(ctx,
            target_memory, layout_constraints, target_regions, result, created)) {
            default_report_failed_instance_creation(task, *constraint.indexes.begin(),
                task.target_proc, target_memory);
        }

        if (created) {
            FieldSpace fs = task.regions[region_indexes[0]].region.get_field_space();
            set<FieldID> s;
            for (FieldID fid: fields) {
                s.insert(fid);
            }
            stringstream msg;
            msg << "Created a colocated instance for task " << task.get_task_name()
                << " for region requirements ";
            for (auto idx: region_indexes) {
               msg << idx << " ";
            }
            msg << "for the following fields "
                << Utilities::to_string(runtime, ctx, fs, s);
            log_mapper.info() << msg.str();
        }

        // mark the region requirements we just did as done
        for (unsigned i=0; i<region_indexes.size(); i++) {
            output.chosen_instances[region_indexes[i]].push_back(result);
            done_regions[region_indexes[i]] = true;
        }
    }
}

void MyMapper::map_task(const MapperContext ctx, const Task &task,
        const MapTaskInput &input, MapTaskOutput &output) {
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
        true/*needs tight bound*/, true/*cache*/, task.target_proc.kind());

    // these are cases this custom mapper do not support, it shouldn't go there
    bool fall_back_to_default_mapper =
        ( chosen.is_inner && ((task.tag & MT_RANK_DISPATCH)!=0) );
    if (fall_back_to_default_mapper) {
        log_mapper.print("Falling back to default mapper in map_task.");
        return DefaultMapper::map_task(ctx, task, input, output);
    }

    // set the variant, task priority, postmap if any and the set of target processors
    output.chosen_variant = chosen.variant;
    output.task_priority = default_policy_select_task_priority(ctx, task);
    output.postmap_task = false;
    default_policy_select_target_processors(ctx, task, output.target_procs);

    /* Look if we can use a cached mapping. */
    CachedMappingPolicy cache_policy = default_policy_select_task_cache_policy(ctx, task);
    const unsigned long long task_hash = compute_task_hash(task);
    pair<TaskID,Processor> cache_key(task.task_id, task.target_proc);
    map< pair<TaskID,Processor>, list<CachedTaskMapping> >::const_iterator
        finder = cached_task_mappings.find(cache_key);
    // This flag says whether we need to recheck the field constraints,
    // possibly because a new field was allocated in a region, so our old
    // cached physical instance(s) is(are) no longer valid
    bool needs_field_constraint_check = false;
    if (cache_policy==DEFAULT_CACHE_POLICY_ENABLE && finder!=cached_task_mappings.end()) {
        bool found = false;
        // Iterate through and see if we can find one with our variant and hash
        for (list<CachedTaskMapping>::const_iterator it=finder->second.begin();
            it != finder->second.end(); it++) {
            if ((it->variant==output.chosen_variant) && (it->task_hash==task_hash)) {
                // Have to copy it before we do the external call which
                // might invalidate our iterator
                output.chosen_instances = it->mapping;
                found = true;
                break;
            }
        }
        if (found) {
            // See if we can acquire these instances still
            if (runtime->acquire_and_filter_instances(ctx, output.chosen_instances)) {
                return;
            }
            // We need to check the constraints here because we had a
            // prior mapping and it failed, which may be the result
            // of a change in the allocated fields of a field space
            needs_field_constraint_check = true;
            // If some of them were deleted, go back and remove this entry
            // Have to renew our iterators since they might have been
            // invalidated during the 'acquire_and_filter_instances' call
            default_remove_cached_task(ctx, output.chosen_variant,
                task_hash, cache_key, output.chosen_instances);
        }
    }

    /* We didn't find a cached version of the mapping so we need to do a full mapping.
     * We already know what variant we want to use.
     * We will do things in this order:
     * - address colocation constraints
     * - address pre-mapping
     * - address each remaining region requirement one by one
     */
    vector<set<FieldID>> missing_fields(task.regions.size());
    runtime->filter_instances(ctx, task, output.chosen_variant,
        output.chosen_instances, missing_fields);
    // Track which regions have already been mapped
    vector<bool> done_regions(task.regions.size(), false);

    // address the colocation constraints
    const ExecutionConstraintSet &constraints =
        runtime->find_execution_constraints(ctx, task.task_id, chosen.variant);
    if (constraints.colocation_constraints.size()>0) {
        map_task_colocation(ctx, task, constraints, output, done_regions);
    }

    // address pre-mapping
    if (!input.premapped_regions.empty()) {
        for (vector<unsigned>::const_iterator it =
            input.premapped_regions.begin(); it !=
            input.premapped_regions.end(); it++) {
            done_regions[*it] = true;
        }
    }

    const TaskLayoutConstraintSet &layout_constraints =
        runtime->find_task_layout_constraints(ctx, task.task_id, output.chosen_variant);
    // Now we need to go through and make instances for any of our
    // regions which do not have space for certain fields
    for (unsigned idx = 0; idx < task.regions.size(); idx++) {
        // already done
        if (done_regions[idx]) {
            continue;
        }
        // empty
        if ((task.regions[idx].privilege == NO_ACCESS) ||
            (task.regions[idx].privilege_fields.empty()) ||
            missing_fields[idx].empty()) {
            continue;
        }

        Memory target_memory = default_policy_select_target_memory(ctx,
            task.target_proc, task.regions[idx]);
        // reduction
        if (task.regions[idx].privilege==REDUCE) {
            size_t footprint;
            if (!default_create_custom_instances(ctx, task.target_proc,
                target_memory, task.regions[idx], idx, missing_fields[idx],
                layout_constraints, needs_field_constraint_check,
                output.chosen_instances[idx], &footprint)) {
                default_report_failed_instance_creation(task, idx,
                    task.target_proc, target_memory, footprint);
            }
            continue;
        }
        // virtual mapping
        if ((task.regions[idx].tag & DefaultMapper::VIRTUAL_MAP)!=0) {
            PhysicalInstance virt_inst = PhysicalInstance::get_virtual_instance();
            output.chosen_instances[idx].push_back(virt_inst);
            continue;
        }
        // already existing valid instances
        vector<PhysicalInstance> valid_instances;
        for (vector<PhysicalInstance>::const_iterator
            it = input.valid_instances[idx].begin(),
            ie = input.valid_instances[idx].end(); it != ie; ++it) {
            if (it->get_location()==target_memory) {
                valid_instances.push_back(*it);
            }
        }
        set<FieldID> valid_missing_fields;
        runtime->filter_instances(ctx, task, idx, output.chosen_variant,
            valid_instances, valid_missing_fields);
#ifndef NDEBUG
        bool check =
#endif
            runtime->acquire_and_filter_instances(ctx, valid_instances);
        assert(check);
        output.chosen_instances[idx] = valid_instances;
        missing_fields[idx] = valid_missing_fields;
        if (missing_fields[idx].empty()) {
            continue;
        }

        // none of the above worked, create an instance

        if ((task.regions[idx].tag & MT_AOS)!=0) {
            // AOS layout requested, treat this region requirement independently
            LayoutConstraintSet lay_constraints;
            vector<DimensionKind> dim_order = {LEGION_DIM_F, LEGION_DIM_X};
            lay_constraints
                .add_constraint(MemoryConstraint(target_memory.kind()))
                .add_constraint(FieldConstraint(task.regions[idx].privilege_fields,
                    true /*contiguous fields*/, true /*fields in order*/))
                .add_constraint(OrderingConstraint(dim_order, false /*contiguous constraints?*/));

            PhysicalInstance result;
            bool created;
            if (!runtime->find_or_create_physical_instance(ctx,
                target_memory, lay_constraints, {task.regions[idx].region}, result, created)) {
                log_mapper.print() << "Could not create instance for region req " << idx
                                   << " which requested AOS layout.";
            }
            if (created) {
                log_mapper.info() << "Created an instance for task " << task.get_task_name()
                                  << " for region requirement " << idx
                                  << " with AOS layout for the following fields: "
                                  << Utilities::to_string(runtime, ctx,
                task.regions[idx].region.get_field_space(), task.regions[idx].privilege_fields);
            }
            output.chosen_instances[idx].push_back(result);
            done_regions[idx] = true;
            continue;
        }

        FieldSpace fs = task.regions[idx].region.get_field_space();
        set<FieldID> s = missing_fields[idx];
        log_mapper.info() << "Trying to create an instance for task " << task.get_task_name()
                          << " for region requirement " << idx
                          << " for the following fields: "
                          << Utilities::to_string(runtime, ctx, fs, s).c_str();
        // log_mapper.print("input.valid_instances[idx].size() = %ld",
        //     input.valid_instances[idx].size());
        // log_mapper.print("output.chosen_instances[idx].size() = %ld",
        //     output.chosen_instances[idx].size());

        size_t footprint;
        if (!default_create_custom_instances(ctx, task.target_proc,
            target_memory, task.regions[idx], idx, missing_fields[idx],
            layout_constraints, needs_field_constraint_check,
            output.chosen_instances[idx], &footprint)) {
            default_report_failed_instance_creation(task, idx,
                task.target_proc, target_memory, footprint);
        }
    }

    if (cache_policy==DEFAULT_CACHE_POLICY_ENABLE) {
        // Now that we are done, let's cache the result so we can use it later
        list<CachedTaskMapping> &map_list = cached_task_mappings[cache_key];
        map_list.push_back(CachedTaskMapping());
        CachedTaskMapping &cached_result = map_list.back();
        cached_result.task_hash = task_hash;
        cached_result.variant = output.chosen_variant;
        cached_result.mapping = output.chosen_instances;
    }
}

#ifdef USE_DCR
void MyMapper::select_sharding_functor(const MapperContext ctx, const Task &task,
        const SelectShardingFunctorInput &input, SelectShardingFunctorOutput &output) {
    log_mapper.spew("Using my sharding");
    if ((task.tag & MT_RANK_DISPATCH)!=0) {
        output.chosen_functor = SFID_RANK_DISPATCH;
    }
    else if ((task.tag & MT_HDF_OUTPUT)!=0) {
        output.chosen_functor = SFID_UNIQUE;
    }
    else {
        output.chosen_functor = SFID_NAIVE;
    }
}
#endif

void MyMapper::default_policy_select_target_processors(MapperContext ctx, const Task &task,
        vector<Processor> &target_procs) {
    DefaultMapper::default_policy_select_target_processors(ctx, task, target_procs);
    // target_procs.push_back(task.target_proc);
}

static void create_mappers(Machine machine, Runtime *runtime,
        const set<Processor> &local_procs) {
    log_mapper.print("Replacing the default mapper by my mapper on all procs.");
    for (set<Processor>::const_iterator it=local_procs.begin(); it!=local_procs.end(); it++) {
        MyMapper *mapper = new MyMapper(runtime, machine, *it, "my_mapper");
        runtime->replace_default_mapper(mapper, *it);
    }
}

void register_mappers() {
    Runtime::add_registration_callback(create_mappers);
#ifdef USE_DCR
    log_mapper.print("Registering my sharding functor for DCR.");
    Runtime::preregister_sharding_functor(SFID_NAIVE, new ShardingNaive());
    Runtime::preregister_sharding_functor(SFID_RANK_DISPATCH, new ShardingRankDispatch());
    Runtime::preregister_sharding_functor(SFID_UNIQUE, new ShardingUnique());
#endif
}

