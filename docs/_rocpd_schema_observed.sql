CREATE TABLE IF NOT EXISTS "rocpd_metadata_00000450_565d_765d_bc41_19a2073d19ee" (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "tag" TEXT NOT NULL,
        "value" TEXT NOT NULL
    );
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE `rocpd_string_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "string" TEXT NOT NULL UNIQUE ON CONFLICT ABORT
    );
CREATE TABLE `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "hash" BIGINT NOT NULL UNIQUE,
        "machine_id" TEXT NOT NULL UNIQUE,
        "system_name" TEXT,
        "hostname" TEXT,
        "release" TEXT,
        "version" TEXT,
        "hardware_name" TEXT,
        "domain_name" TEXT
    );
CREATE TABLE `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "ppid" INTEGER,
        "pid" INTEGER NOT NULL,
        "init" BIGINT,
        "fini" BIGINT,
        "start" BIGINT,
        "end" BIGINT,
        "command" TEXT,
        "environment" JSONB DEFAULT "{}" NOT NULL,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_info_thread_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "ppid" INTEGER,
        "pid" INTEGER NOT NULL,
        "tid" INTEGER NOT NULL,
        "name" TEXT,
        "start" BIGINT,
        "end" BIGINT,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_info_agent_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "type" TEXT CHECK ("type" IN ('CPU', 'GPU')),
        "absolute_index" INTEGER,
        "logical_index" INTEGER,
        "type_index" INTEGER,
        "uuid" INTEGER,
        "name" TEXT,
        "model_name" TEXT,
        "vendor_name" TEXT,
        "product_name" TEXT,
        "user_name" TEXT,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_info_queue_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "name" TEXT,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_info_stream_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "name" TEXT,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_info_pmc_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "agent_id" INTEGER,
        "target_arch" TEXT CHECK ("target_arch" IN ('CPU', 'GPU')),
        "event_code" INT,
        "instance_id" INTEGER,
        "name" TEXT NOT NULL,
        "symbol" TEXT NOT NULL,
        "description" TEXT,
        "long_description" TEXT DEFAULT "",
        "component" TEXT,
        "units" TEXT DEFAULT "",
        "value_type" TEXT CHECK ("value_type" IN ('ABS', 'ACCUM', 'RELATIVE')),
        "block" TEXT,
        "expression" TEXT,
        "is_constant" INTEGER,
        "is_derived" INTEGER,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (agent_id) REFERENCES `rocpd_info_agent_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_info_code_object_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "agent_id" INTEGER,
        "uri" TEXT,
        "load_base" BIGINT,
        "load_size" BIGINT,
        "load_delta" BIGINT,
        "storage_type" TEXT CHECK ("storage_type" IN ('FILE', 'MEMORY')),
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (agent_id) REFERENCES `rocpd_info_agent_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_info_kernel_symbol_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "code_object_id" INTEGER NOT NULL,
        "kernel_name" TEXT,
        "display_name" TEXT,
        "kernel_object" INTEGER,
        "kernarg_segment_size" INTEGER,
        "kernarg_segment_alignment" INTEGER,
        "group_segment_size" INTEGER,
        "private_segment_size" INTEGER,
        "sgpr_count" INTEGER,
        "arch_vgpr_count" INTEGER,
        "accum_vgpr_count" INTEGER,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (code_object_id) REFERENCES `rocpd_info_code_object_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_track_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER,
        "tid" INTEGER,
        "name_id" INTEGER,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (tid) REFERENCES `rocpd_info_thread_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (name_id) REFERENCES `rocpd_string_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "category_id" INTEGER,
        "stack_id" INTEGER,
        "parent_stack_id" INTEGER,
        "correlation_id" INTEGER,
        "call_stack" JSONB DEFAULT "{}" NOT NULL,
        "line_info" JSONB DEFAULT "{}" NOT NULL,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (category_id) REFERENCES `rocpd_string_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_arg_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "event_id" INTEGER NOT NULL,
        "position" INTEGER NOT NULL,
        "type" TEXT NOT NULL,
        "name" TEXT NOT NULL,
        "value" TEXT, -- TODO: discuss make it value_id and integer, refer to string table --
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_pmc_event_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "event_id" INTEGER,
        "pmc_id" INTEGER NOT NULL,
        "value" REAL DEFAULT 0.0,
        "extdata" JSONB DEFAULT "{}",
        FOREIGN KEY (pmc_id) REFERENCES `rocpd_info_pmc_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_region_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "tid" INTEGER NOT NULL,
        "start" BIGINT NOT NULL,
        "end" BIGINT NOT NULL,
        "name_id" INTEGER NOT NULL,
        "event_id" INTEGER,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (tid) REFERENCES `rocpd_info_thread_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (name_id) REFERENCES `rocpd_string_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_sample_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "track_id" INTEGER NOT NULL,
        "timestamp" BIGINT NOT NULL,
        "event_id" INTEGER,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (track_id) REFERENCES `rocpd_track_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_kernel_dispatch_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "tid" INTEGER,
        "agent_id" INTEGER NOT NULL,
        "kernel_id" INTEGER NOT NULL,
        "dispatch_id" INTEGER NOT NULL,
        "queue_id" INTEGER NOT NULL,
        "stream_id" INTEGER NOT NULL,
        "start" BIGINT NOT NULL,
        "end" BIGINT NOT NULL,
        "private_segment_size" INTEGER,
        "group_segment_size" INTEGER,
        "workgroup_size_x" INTEGER NOT NULL,
        "workgroup_size_y" INTEGER NOT NULL,
        "workgroup_size_z" INTEGER NOT NULL,
        "grid_size_x" INTEGER NOT NULL,
        "grid_size_y" INTEGER NOT NULL,
        "grid_size_z" INTEGER NOT NULL,
        "region_name_id" INTEGER,
        "event_id" INTEGER,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (tid) REFERENCES `rocpd_info_thread_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (agent_id) REFERENCES `rocpd_info_agent_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (kernel_id) REFERENCES `rocpd_info_kernel_symbol_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (queue_id) REFERENCES `rocpd_info_queue_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (stream_id) REFERENCES `rocpd_info_stream_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (region_name_id) REFERENCES `rocpd_string_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_memory_copy_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "tid" INTEGER,
        "start" BIGINT NOT NULL,
        "end" BIGINT NOT NULL,
        "name_id" INTEGER NOT NULL,
        "dst_agent_id" INTEGER,
        "dst_address" INTEGER,
        "src_agent_id" INTEGER,
        "src_address" INTEGER,
        "size" INTEGER NOT NULL,
        "queue_id" INTEGER,
        "stream_id" INTEGER,
        "region_name_id" INTEGER,
        "event_id" INTEGER,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (tid) REFERENCES `rocpd_info_thread_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (name_id) REFERENCES `rocpd_string_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (dst_agent_id) REFERENCES `rocpd_info_agent_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (src_agent_id) REFERENCES `rocpd_info_agent_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (stream_id) REFERENCES `rocpd_info_stream_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (queue_id) REFERENCES `rocpd_info_queue_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (region_name_id) REFERENCES `rocpd_string_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE TABLE `rocpd_memory_allocate_00000450_565d_765d_bc41_19a2073d19ee` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid" TEXT DEFAULT "00000450-565d-765d-bc41-19a2073d19ee" NOT NULL,
        "nid" INTEGER NOT NULL,
        "pid" INTEGER NOT NULL,
        "tid" INTEGER,
        "agent_id" INTEGER,
        "type" TEXT CHECK ("type" IN ('ALLOC', 'FREE', 'REALLOC', 'RECLAIM')),
        "level" TEXT CHECK ("level" IN ('REAL', 'VIRTUAL', 'SCRATCH')),
        "start" BIGINT NOT NULL,
        "end" BIGINT NOT NULL,
        "address" INTEGER,
        "size" INTEGER NOT NULL,
        "queue_id" INTEGER,
        "stream_id" INTEGER,
        "event_id" INTEGER,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (tid) REFERENCES `rocpd_info_thread_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (agent_id) REFERENCES `rocpd_info_agent_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (stream_id) REFERENCES `rocpd_info_stream_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (queue_id) REFERENCES `rocpd_info_queue_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee` (id) ON UPDATE CASCADE
    );
CREATE VIEW `rocpd_metadata` AS
SELECT
    *
FROM
    `rocpd_metadata_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_metadata(id,tag,value) */;
CREATE VIEW `rocpd_string` AS
SELECT
    *
FROM
    `rocpd_string_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_string(id,guid,string) */;
CREATE VIEW `rocpd_info_node` AS
SELECT
    *
FROM
    `rocpd_info_node_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_node(id,guid,hash,machine_id,system_name,hostname,"release",version,hardware_name,domain_name) */;
CREATE VIEW `rocpd_info_process` AS
SELECT
    *
FROM
    `rocpd_info_process_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_process(id,guid,nid,ppid,pid,init,fini,start,"end",command,environment,extdata) */;
CREATE VIEW `rocpd_info_thread` AS
SELECT
    *
FROM
    `rocpd_info_thread_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_thread(id,guid,nid,ppid,pid,tid,name,start,"end",extdata) */;
CREATE VIEW `rocpd_info_agent` AS
SELECT
    *
FROM
    `rocpd_info_agent_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_agent(id,guid,nid,pid,type,absolute_index,logical_index,type_index,uuid,name,model_name,vendor_name,product_name,user_name,extdata) */;
CREATE VIEW `rocpd_info_queue` AS
SELECT
    *
FROM
    `rocpd_info_queue_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_queue(id,guid,nid,pid,name,extdata) */;
CREATE VIEW `rocpd_info_stream` AS
SELECT
    *
FROM
    `rocpd_info_stream_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_stream(id,guid,nid,pid,name,extdata) */;
CREATE VIEW `rocpd_info_pmc` AS
SELECT
    *
FROM
    `rocpd_info_pmc_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_pmc(id,guid,nid,pid,agent_id,target_arch,event_code,instance_id,name,symbol,description,long_description,component,units,value_type,block,expression,is_constant,is_derived,extdata) */;
CREATE VIEW `rocpd_info_code_object` AS
SELECT
    *
FROM
    `rocpd_info_code_object_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_code_object(id,guid,nid,pid,agent_id,uri,load_base,load_size,load_delta,storage_type,extdata) */;
CREATE VIEW `rocpd_info_kernel_symbol` AS
SELECT
    *
FROM
    `rocpd_info_kernel_symbol_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_info_kernel_symbol(id,guid,nid,pid,code_object_id,kernel_name,display_name,kernel_object,kernarg_segment_size,kernarg_segment_alignment,group_segment_size,private_segment_size,sgpr_count,arch_vgpr_count,accum_vgpr_count,extdata) */;
CREATE VIEW `rocpd_track` AS
SELECT
    *
FROM
    `rocpd_track_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_track(id,guid,nid,pid,tid,name_id,extdata) */;
CREATE VIEW `rocpd_event` AS
SELECT
    *
FROM
    `rocpd_event_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_event(id,guid,category_id,stack_id,parent_stack_id,correlation_id,call_stack,line_info,extdata) */;
CREATE VIEW `rocpd_arg` AS
SELECT
    *
FROM
    `rocpd_arg_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_arg(id,guid,event_id,position,type,name,value,extdata) */;
CREATE VIEW `rocpd_pmc_event` AS
SELECT
    *
FROM
    `rocpd_pmc_event_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_pmc_event(id,guid,event_id,pmc_id,value,extdata) */;
CREATE VIEW `rocpd_region` AS
SELECT
    *
FROM
    `rocpd_region_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_region(id,guid,nid,pid,tid,start,"end",name_id,event_id,extdata) */;
CREATE VIEW `rocpd_sample` AS
SELECT
    *
FROM
    `rocpd_sample_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_sample(id,guid,track_id,timestamp,event_id,extdata) */;
CREATE VIEW `rocpd_kernel_dispatch` AS
SELECT
    *
FROM
    `rocpd_kernel_dispatch_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_kernel_dispatch(id,guid,nid,pid,tid,agent_id,kernel_id,dispatch_id,queue_id,stream_id,start,"end",private_segment_size,group_segment_size,workgroup_size_x,workgroup_size_y,workgroup_size_z,grid_size_x,grid_size_y,grid_size_z,region_name_id,event_id,extdata) */;
CREATE VIEW `rocpd_memory_copy` AS
SELECT
    *
FROM
    `rocpd_memory_copy_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_memory_copy(id,guid,nid,pid,tid,start,"end",name_id,dst_agent_id,dst_address,src_agent_id,src_address,size,queue_id,stream_id,region_name_id,event_id,extdata) */;
CREATE VIEW `rocpd_memory_allocate` AS
SELECT
    *
FROM
    `rocpd_memory_allocate_00000450_565d_765d_bc41_19a2073d19ee`
/* rocpd_memory_allocate(id,guid,nid,pid,tid,agent_id,type,level,start,"end",address,size,queue_id,stream_id,event_id,extdata) */;
CREATE VIEW `code_objects` AS
SELECT
    CO.id,
    CO.guid,
    CO.nid,
    P.pid,
    A.absolute_index AS agent_abs_index,
    CO.uri,
    CO.load_base,
    CO.load_size,
    CO.load_delta,
    CO.storage_type AS storage_type_str,
    JSON_EXTRACT(CO.extdata, '$.size') AS code_object_size,
    JSON_EXTRACT(CO.extdata, '$.storage_type') AS storage_type,
    JSON_EXTRACT(CO.extdata, '$.memory_base') AS memory_base,
    JSON_EXTRACT(CO.extdata, '$.memory_size') AS memory_size
FROM
    `rocpd_info_code_object` CO
    INNER JOIN `rocpd_info_agent` A ON CO.agent_id = A.id
    AND CO.guid = A.guid
    INNER JOIN `rocpd_info_process` P ON CO.pid = P.id
    AND CO.guid = P.guid
/* code_objects(id,guid,nid,pid,agent_abs_index,uri,load_base,load_size,load_delta,storage_type_str,code_object_size,storage_type,memory_base,memory_size) */;
CREATE VIEW `kernel_symbols` AS
SELECT
    KS.id,
    KS.guid,
    KS.nid,
    P.pid,
    KS.code_object_id,
    KS.kernel_name,
    KS.display_name,
    KS.kernel_object,
    KS.kernarg_segment_size,
    KS.kernarg_segment_alignment,
    KS.group_segment_size,
    KS.private_segment_size,
    KS.sgpr_count,
    KS.arch_vgpr_count,
    KS.accum_vgpr_count,
    JSON_EXTRACT(KS.extdata, '$.size') AS kernel_symbol_size,
    JSON_EXTRACT(KS.extdata, '$.kernel_id') AS kernel_id,
    JSON_EXTRACT(KS.extdata, '$.kernel_code_entry_byte_offset') AS kernel_code_entry_byte_offset,
    JSON_EXTRACT(KS.extdata, '$.formatted_kernel_name') AS formatted_kernel_name,
    JSON_EXTRACT(KS.extdata, '$.demangled_kernel_name') AS demangled_kernel_name,
    JSON_EXTRACT(KS.extdata, '$.truncated_kernel_name') AS truncated_kernel_name,
    JSON_EXTRACT(KS.extdata, '$.kernel_address.handle') AS kernel_address
FROM
    `rocpd_info_kernel_symbol` KS
    INNER JOIN `rocpd_info_process` P ON KS.pid = P.id
    AND KS.guid = P.guid
/* kernel_symbols(id,guid,nid,pid,code_object_id,kernel_name,display_name,kernel_object,kernarg_segment_size,kernarg_segment_alignment,group_segment_size,private_segment_size,sgpr_count,arch_vgpr_count,accum_vgpr_count,kernel_symbol_size,kernel_id,kernel_code_entry_byte_offset,formatted_kernel_name,demangled_kernel_name,truncated_kernel_name,kernel_address) */;
CREATE VIEW `processes` AS
SELECT
    N.id AS nid,
    N.machine_id,
    N.system_name,
    N.hostname,
    N.release AS system_release,
    N.version AS system_version,
    P.guid,
    P.ppid,
    P.pid,
    P.init,
    P.start,
    P.end,
    P.fini,
    P.command
FROM
    `rocpd_info_process` P
    INNER JOIN `rocpd_info_node` N ON N.id = P.nid
    AND N.guid = P.guid
/* processes(nid,machine_id,system_name,hostname,system_release,system_version,guid,ppid,pid,init,start,"end",fini,command) */;
CREATE VIEW `threads` AS
SELECT
    N.id AS nid,
    N.machine_id,
    N.system_name,
    N.hostname,
    N.release AS system_release,
    N.version AS system_version,
    P.guid,
    P.ppid,
    P.pid,
    T.tid,
    T.start,
    T.end,
    T.name
FROM
    `rocpd_info_thread` T
    INNER JOIN `rocpd_info_process` P ON P.id = T.pid
    AND N.guid = T.guid
    INNER JOIN `rocpd_info_node` N ON N.id = T.nid
    AND N.guid = T.guid
/* threads(nid,machine_id,system_name,hostname,system_release,system_version,guid,ppid,pid,tid,start,"end",name) */;
CREATE VIEW `regions` AS
SELECT
    R.id,
    R.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    S.string AS name,
    R.nid,
    P.pid,
    T.tid,
    R.start,
    R.end,
    (R.end - R.start) AS duration,
    R.event_id,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id,
    E.extdata,
    E.call_stack,
    E.line_info
FROM
    `rocpd_region` R
    INNER JOIN `rocpd_event` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `rocpd_string` S ON S.id = R.name_id
    AND S.guid = R.guid
    INNER JOIN `rocpd_info_process` P ON P.id = R.pid
    AND P.guid = R.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = R.tid
    AND T.guid = R.guid
/* regions(id,guid,category,name,nid,pid,tid,start,"end",duration,event_id,stack_id,parent_stack_id,corr_id,extdata,call_stack,line_info) */;
CREATE VIEW `region_args` AS
SELECT
    R.id,
    R.guid,
    R.nid,
    P.pid,
    A.type,
    A.name,
    A.value
FROM
    `rocpd_region` R
    INNER JOIN `rocpd_event` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `rocpd_arg` A ON A.event_id = E.id
    AND A.guid = R.guid
    INNER JOIN `rocpd_info_process` P ON P.id = R.pid
    AND P.guid = R.guid
/* region_args(id,guid,nid,pid,type,name,value) */;
CREATE VIEW `samples` AS
SELECT
    R.id,
    R.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = T.name_id
            AND RS.guid = T.guid
    ) AS name,
    T.nid,
    P.pid,
    TH.tid,
    R.timestamp,
    R.event_id,
    E.stack_id AS stack_id,
    E.parent_stack_id AS parent_stack_id,
    E.correlation_id AS corr_id,
    E.extdata AS extdata,
    E.call_stack AS call_stack,
    E.line_info AS line_info
FROM
    `rocpd_sample` R
    INNER JOIN `rocpd_track` T ON T.id = R.track_id
    AND T.guid = R.guid
    INNER JOIN `rocpd_event` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `rocpd_info_process` P ON P.id = T.pid
    AND P.guid = T.guid
    INNER JOIN `rocpd_info_thread` TH ON TH.id = T.tid
    AND TH.guid = T.guid
/* samples(id,guid,category,name,nid,pid,tid,timestamp,event_id,stack_id,parent_stack_id,corr_id,extdata,call_stack,line_info) */;
CREATE VIEW `sample_regions` AS
SELECT
    R.id,
    R.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = T.name_id
            AND RS.guid = T.guid
    ) AS name,
    T.nid,
    P.pid,
    TH.tid,
    R.timestamp AS start,
    R.timestamp AS END,
    (R.timestamp - R.timestamp) AS duration,
    R.event_id,
    E.stack_id AS stack_id,
    E.parent_stack_id AS parent_stack_id,
    E.correlation_id AS corr_id,
    E.extdata AS extdata,
    E.call_stack AS call_stack,
    E.line_info AS line_info
FROM
    `rocpd_sample` R
    INNER JOIN `rocpd_track` T ON T.id = R.track_id
    AND T.guid = R.guid
    INNER JOIN `rocpd_event` E ON E.id = R.event_id
    AND E.guid = R.guid
    INNER JOIN `rocpd_info_process` P ON P.id = T.pid
    AND P.guid = T.guid
    INNER JOIN `rocpd_info_thread` TH ON TH.id = T.tid
    AND TH.guid = T.guid
/* sample_regions(id,guid,category,name,nid,pid,tid,start,"END",duration,event_id,stack_id,parent_stack_id,corr_id,extdata,call_stack,line_info) */;
CREATE VIEW `regions_and_samples` AS
SELECT
    *
FROM
    `regions`
UNION ALL
SELECT
    *
FROM
    `sample_regions`
/* regions_and_samples(id,guid,category,name,nid,pid,tid,start,"end",duration,event_id,stack_id,parent_stack_id,corr_id,extdata,call_stack,line_info) */;
CREATE VIEW `kernels` AS
SELECT
    K.id,
    K.guid,
    T.tid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    R.string AS region,
    S.display_name AS name,
    K.nid,
    P.pid,
    A.absolute_index AS agent_abs_index,
    A.logical_index AS agent_log_index,
    A.type_index AS agent_type_index,
    A.type AS agent_type,
    S.code_object_id AS code_object_id,
    K.kernel_id,
    K.dispatch_id,
    K.stream_id,
    K.queue_id,
    Q.name AS queue,
    ST.name AS stream,
    K.start,
    K.end,
    (K.end - K.start) AS duration,
    K.grid_size_x AS grid_x,
    K.grid_size_y AS grid_y,
    K.grid_size_z AS grid_z,
    K.workgroup_size_x AS workgroup_x,
    K.workgroup_size_y AS workgroup_y,
    K.workgroup_size_z AS workgroup_z,
    K.group_segment_size AS lds_size,
    K.private_segment_size AS scratch_size,
    S.arch_vgpr_count AS vgpr_count,
    S.accum_vgpr_count,
    S.sgpr_count,
    S.group_segment_size AS static_lds_size,
    S.private_segment_size AS static_scratch_size,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id
FROM
    `rocpd_kernel_dispatch` K
    INNER JOIN `rocpd_info_agent` A ON A.id = K.agent_id
    AND A.guid = K.guid
    INNER JOIN `rocpd_event` E ON E.id = K.event_id
    AND E.guid = K.guid
    INNER JOIN `rocpd_string` R ON R.id = K.region_name_id
    AND R.guid = K.guid
    INNER JOIN `rocpd_info_kernel_symbol` S ON S.id = K.kernel_id
    AND S.guid = K.guid
    LEFT JOIN `rocpd_info_stream` ST ON ST.id = K.stream_id
    AND ST.guid = K.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = K.queue_id
    AND Q.guid = K.guid
    INNER JOIN `rocpd_info_process` P ON P.id = Q.pid
    AND P.guid = Q.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = K.tid
    AND T.guid = K.guid
/* kernels(id,guid,tid,category,region,name,nid,pid,agent_abs_index,agent_log_index,agent_type_index,agent_type,code_object_id,kernel_id,dispatch_id,stream_id,queue_id,queue,stream,start,"end",duration,grid_x,grid_y,grid_z,workgroup_x,workgroup_y,workgroup_z,lds_size,scratch_size,vgpr_count,accum_vgpr_count,sgpr_count,static_lds_size,static_scratch_size,stack_id,parent_stack_id,corr_id) */;
CREATE VIEW `pmc_info` AS
SELECT
    PMC_I.id,
    PMC_I.guid,
    PMC_I.nid,
    P.pid,
    A.absolute_index AS agent_abs_index,
    PMC_I.is_constant,
    PMC_I.is_derived,
    PMC_I.name,
    PMC_I.description,
    PMC_I.block,
    PMC_I.expression
FROM
    `rocpd_info_pmc` PMC_I
    INNER JOIN `rocpd_info_agent` A ON PMC_I.agent_id = A.id
    AND PMC_I.guid = A.guid
    INNER JOIN `rocpd_info_process` P ON P.id = PMC_I.pid
    AND PMC_I.guid = P.guid
/* pmc_info(id,guid,nid,pid,agent_abs_index,is_constant,is_derived,name,description,block,expression) */;
CREATE VIEW `pmc_events` AS
SELECT
    PMC_E.id,
    PMC_E.guid,
    PMC_E.pmc_id,
    E.id AS event_id,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    (
        SELECT
            display_name
        FROM
            `rocpd_info_kernel_symbol` KS
        WHERE
            KS.id = K.kernel_id
            AND KS.guid = K.guid
    ) AS name,
    K.nid,
    P.pid,
    K.dispatch_id,
    K.start,
    K.end,
    (K.end - K.start) AS duration,
    PMC_I.name AS counter_name,
    PMC_E.value AS counter_value
FROM
    `rocpd_pmc_event` PMC_E
    INNER JOIN `rocpd_info_pmc` PMC_I ON PMC_I.id = PMC_E.pmc_id
    AND PMC_I.guid = PMC_E.guid
    INNER JOIN `rocpd_event` E ON E.id = PMC_E.event_id
    AND E.guid = PMC_E.guid
    INNER JOIN `rocpd_kernel_dispatch` K ON K.event_id = PMC_E.event_id
    AND K.guid = PMC_E.guid
    INNER JOIN `rocpd_info_process` P ON P.id = K.pid
    AND P.guid = K.guid
/* pmc_events(id,guid,pmc_id,event_id,category,name,nid,pid,dispatch_id,start,"end",duration,counter_name,counter_value) */;
CREATE VIEW `events_args` AS
SELECT
    E.id AS event_id,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id,
    A.position AS arg_position,
    A.type AS arg_type,
    A.name AS arg_name,
    A.value AS arg_value,
    E.call_stack,
    E.line_info,
    A.extdata
FROM
    `rocpd_event` E
    INNER JOIN `rocpd_arg` A ON A.event_id = E.id
    AND A.guid = E.guid
/* events_args(event_id,category,stack_id,parent_stack_id,correlation_id,arg_position,arg_type,arg_name,arg_value,call_stack,line_info,extdata) */;
CREATE VIEW `stream_args` AS
SELECT
    A.id AS argument_id,
    A.event_id AS event_id,
    A.position AS arg_position,
    A.type AS arg_type,
    A.value AS arg_value,
    JSON_EXTRACT(A.extdata, '$.stream_id') AS stream_id,
    S.nid,
    P.pid,
    S.name AS stream_name,
    S.extdata AS extdata
FROM
    `rocpd_arg` A
    INNER JOIN `rocpd_info_stream` S ON JSON_EXTRACT(A.extdata, '$.stream_id') = S.id
    AND A.guid = S.guid
    INNER JOIN `rocpd_info_process` P ON P.id = S.pid
    AND P.guid = S.guid
WHERE
    A.name = 'stream'
/* stream_args(argument_id,event_id,arg_position,arg_type,arg_value,stream_id,nid,pid,stream_name,extdata) */;
CREATE VIEW `memory_copies` AS
SELECT
    M.id,
    M.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    M.nid,
    P.pid,
    T.tid,
    M.start,
    M.end,
    (M.end - M.start) AS duration,
    S.string AS name,
    R.string AS region_name,
    M.stream_id,
    M.queue_id,
    ST.name AS stream_name,
    Q.name AS queue_name,
    M.size,
    dst_agent.name AS dst_device,
    dst_agent.absolute_index AS dst_agent_abs_index,
    dst_agent.logical_index AS dst_agent_log_index,
    dst_agent.type_index AS dst_agent_type_index,
    dst_agent.type AS dst_agent_type,
    M.dst_address,
    src_agent.name AS src_device,
    src_agent.absolute_index AS src_agent_abs_index,
    src_agent.logical_index AS src_agent_log_index,
    src_agent.type_index AS src_agent_type_index,
    src_agent.type AS src_agent_type,
    M.src_address,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id
FROM
    `rocpd_memory_copy` M
    INNER JOIN `rocpd_string` S ON S.id = M.name_id
    AND S.guid = M.guid
    LEFT JOIN `rocpd_string` R ON R.id = M.region_name_id
    AND R.guid = M.guid
    INNER JOIN `rocpd_info_agent` dst_agent ON dst_agent.id = M.dst_agent_id
    AND dst_agent.guid = M.guid
    INNER JOIN `rocpd_info_agent` src_agent ON src_agent.id = M.src_agent_id
    AND src_agent.guid = M.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = M.queue_id
    AND Q.guid = M.guid
    LEFT JOIN `rocpd_info_stream` ST ON ST.id = M.stream_id
    AND ST.guid = M.guid
    INNER JOIN `rocpd_event` E ON E.id = M.event_id
    AND E.guid = M.guid
    INNER JOIN `rocpd_info_process` P ON P.id = M.pid
    AND P.guid = M.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = M.tid
    AND T.guid = M.guid
/* memory_copies(id,guid,category,nid,pid,tid,start,"end",duration,name,region_name,stream_id,queue_id,stream_name,queue_name,size,dst_device,dst_agent_abs_index,dst_agent_log_index,dst_agent_type_index,dst_agent_type,dst_address,src_device,src_agent_abs_index,src_agent_log_index,src_agent_type_index,src_agent_type,src_address,stack_id,parent_stack_id,corr_id) */;
CREATE VIEW `memory_allocations` AS
SELECT
    M.id,
    M.guid,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    M.nid,
    P.pid,
    T.tid,
    M.start,
    M.end,
    (M.end - M.start) AS duration,
    M.type,
    M.level,
    A.name AS agent_name,
    A.absolute_index AS agent_abs_index,
    A.logical_index AS agent_log_index,
    A.type_index AS agent_type_index,
    A.type AS agent_type,
    M.address,
    M.size,
    M.queue_id,
    Q.name AS queue_name,
    M.stream_id,
    ST.name AS stream_name,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id
FROM
    `rocpd_memory_allocate` M
    LEFT JOIN `rocpd_info_agent` A ON M.agent_id = A.id
    AND M.guid = A.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = M.queue_id
    AND Q.guid = M.guid
    LEFT JOIN `rocpd_info_stream` ST ON ST.id = M.stream_id
    AND ST.guid = M.guid
    INNER JOIN `rocpd_event` E ON E.id = M.event_id
    AND E.guid = M.guid
    INNER JOIN `rocpd_info_process` P ON P.id = M.pid
    AND P.guid = M.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = M.tid
    AND P.guid = M.guid
/* memory_allocations(id,guid,category,nid,pid,tid,start,"end",duration,type,level,agent_name,agent_abs_index,agent_log_index,agent_type_index,agent_type,address,size,queue_id,queue_name,stream_id,stream_name,stack_id,parent_stack_id,corr_id) */;
CREATE VIEW `scratch_memory` AS
SELECT
    M.id,
    M.guid,
    M.nid,
    P.pid,
    M.type AS operation,
    A.name AS agent_name,
    A.absolute_index AS agent_abs_index,
    A.logical_index AS agent_log_index,
    A.type_index AS agent_type_index,
    A.type AS agent_type,
    M.queue_id,
    T.tid,
    JSON_EXTRACT(M.extdata, '$.flags') AS alloc_flags,
    M.start,
    M.end,
    (M.end - M.start) AS duration,
    M.size,
    M.address,
    E.correlation_id,
    E.stack_id,
    E.parent_stack_id,
    E.correlation_id AS corr_id,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    E.extdata AS event_extdata
FROM
    `rocpd_memory_allocate` M
    LEFT JOIN `rocpd_info_agent` A ON M.agent_id = A.id
    AND M.guid = A.guid
    LEFT JOIN `rocpd_info_queue` Q ON Q.id = M.queue_id
    AND Q.guid = M.guid
    INNER JOIN `rocpd_event` E ON E.id = M.event_id
    AND E.guid = M.guid
    INNER JOIN `rocpd_info_process` P ON P.id = M.pid
    AND P.guid = M.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = M.tid
    AND T.guid = M.guid
WHERE
    M.level = 'SCRATCH'
ORDER BY
    M.start ASC
/* scratch_memory(id,guid,nid,pid,operation,agent_name,agent_abs_index,agent_log_index,agent_type_index,agent_type,queue_id,tid,alloc_flags,start,"end",duration,size,address,correlation_id,stack_id,parent_stack_id,corr_id,category,event_extdata) */;
CREATE VIEW `counters_collection` AS
SELECT
    MIN(PMC_E.id) AS id,
    PMC_E.guid,
    K.dispatch_id,
    K.kernel_id,
    E.id AS event_id,
    E.correlation_id,
    E.stack_id,
    E.parent_stack_id,
    P.pid,
    T.tid,
    K.agent_id,
    A.absolute_index AS agent_abs_index,
    A.logical_index AS agent_log_index,
    A.type_index AS agent_type_index,
    A.type AS agent_type,
    K.queue_id,
    k.grid_size_x AS grid_size_x,
    k.grid_size_y AS grid_size_y,
    k.grid_size_z AS grid_size_z,
    (K.grid_size_x * K.grid_size_y * K.grid_size_z) AS grid_size,
    S.display_name AS kernel_name,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = K.region_name_id
            AND RS.guid = K.guid
    ) AS kernel_region,
    K.workgroup_size_x AS workgroup_size_x,
    K.workgroup_size_y AS workgroup_size_y,
    K.workgroup_size_z AS workgroup_size_z,
    (K.workgroup_size_x * K.workgroup_size_y * K.workgroup_size_z) AS workgroup_size,
    K.group_segment_size AS lds_block_size,
    K.private_segment_size AS scratch_size,
    S.arch_vgpr_count AS vgpr_count,
    S.accum_vgpr_count,
    S.sgpr_count,
    PMC_I.name AS counter_name,
    PMC_I.symbol AS counter_symbol,
    PMC_I.component,
    PMC_I.description,
    PMC_I.block,
    PMC_I.expression,
    PMC_I.value_type,
    PMC_I.id AS counter_id,
    SUM(PMC_E.value) AS value,
    K.start,
    K.end,
    PMC_I.is_constant,
    PMC_I.is_derived,
    (K.end - K.start) AS duration,
    (
        SELECT
            string
        FROM
            `rocpd_string` RS
        WHERE
            RS.id = E.category_id
            AND RS.guid = E.guid
    ) AS category,
    K.nid,
    E.extdata,
    S.code_object_id
FROM
    `rocpd_pmc_event` PMC_E
    INNER JOIN `rocpd_info_pmc` PMC_I ON PMC_I.id = PMC_E.pmc_id
    AND PMC_I.guid = PMC_E.guid
    INNER JOIN `rocpd_event` E ON E.id = PMC_E.event_id
    AND E.guid = PMC_E.guid
    INNER JOIN `rocpd_kernel_dispatch` K ON K.event_id = PMC_E.event_id
    AND K.guid = PMC_E.guid
    INNER JOIN `rocpd_info_agent` A ON A.id = K.agent_id
    AND A.guid = K.guid
    INNER JOIN `rocpd_info_kernel_symbol` S ON S.id = K.kernel_id
    AND S.guid = K.guid
    INNER JOIN `rocpd_info_process` P ON P.id = K.pid
    AND P.guid = K.guid
    INNER JOIN `rocpd_info_thread` T ON T.id = K.tid
    AND T.guid = K.guid
GROUP BY
    PMC_E.guid,
    K.dispatch_id,
    PMC_I.name,
    K.agent_id
/* counters_collection(id,guid,dispatch_id,kernel_id,event_id,correlation_id,stack_id,parent_stack_id,pid,tid,agent_id,agent_abs_index,agent_log_index,agent_type_index,agent_type,queue_id,grid_size_x,grid_size_y,grid_size_z,grid_size,kernel_name,kernel_region,workgroup_size_x,workgroup_size_y,workgroup_size_z,workgroup_size,lds_block_size,scratch_size,vgpr_count,accum_vgpr_count,sgpr_count,counter_name,counter_symbol,component,description,block,expression,value_type,counter_id,value,start,"end",is_constant,is_derived,duration,category,nid,extdata,code_object_id) */;
CREATE VIEW `top_kernels` AS
SELECT
    S.display_name AS name,
    COUNT(K.kernel_id) AS total_calls,
    SUM(K.end - K.start) / 1000.0 AS total_duration,
    (SUM(K.end - K.start) / COUNT(K.kernel_id)) / 1000.0 AS average,
    SUM(K.end - K.start) * 100.0 / (
        SELECT
            SUM(A.end - A.start)
        FROM
            `rocpd_kernel_dispatch` A
    ) AS percentage
FROM
    `rocpd_kernel_dispatch` K
    INNER JOIN `rocpd_info_kernel_symbol` S ON S.id = K.kernel_id
    AND S.guid = K.guid
GROUP BY
    name
ORDER BY
    total_duration DESC
/* top_kernels(name,total_calls,total_duration,average,percentage) */;
CREATE VIEW `busy` AS
SELECT
    A.agent_id,
    AG.type,
    GpuTime,
    WallTime,
    GpuTime * 1.0 / WallTime AS Busy
FROM
    (
        SELECT
            agent_id,
            guid,
            SUM(END - start) AS GpuTime
        FROM
            (
                SELECT
                    agent_id,
                    guid,
                    END,
                    start
                FROM
                    `rocpd_kernel_dispatch`
                UNION ALL
                SELECT
                    dst_agent_id AS agent_id,
                    guid,
                    END,
                    start
                FROM
                    `rocpd_memory_copy`
            )
        GROUP BY
            agent_id,
            guid
    ) A
    INNER JOIN (
        SELECT
            MAX(END) - MIN(start) AS WallTime
        FROM
            (
                SELECT
                    END,
                    start
                FROM
                    `rocpd_kernel_dispatch`
                UNION ALL
                SELECT
                    END,
                    start
                FROM
                    `rocpd_memory_copy`
            )
    ) W ON 1 = 1
    INNER JOIN `rocpd_info_agent` AG ON AG.id = A.agent_id
    AND AG.guid = A.guid
/* busy(agent_id,type,GpuTime,WallTime,Busy) */;
CREATE VIEW `top` AS
SELECT
    name,
    COUNT(*) AS total_calls,
    SUM(duration) / 1000.0 AS total_duration,
    (SUM(duration) / COUNT(*)) / 1000.0 AS average,
    SUM(duration) * 100.0 / total_time AS percentage
FROM
    (
        -- Kernel operations
        SELECT
            ks.display_name AS name,
            (kd.end - kd.start) AS duration
        FROM
            `rocpd_kernel_dispatch` kd
            INNER JOIN `rocpd_info_kernel_symbol` ks ON kd.kernel_id = ks.id
            AND kd.guid = ks.guid
        UNION ALL
        -- Memory operations
        SELECT
            rs.string AS name,
            (END - start) AS duration
        FROM
            `rocpd_memory_copy` mc
            INNER JOIN `rocpd_string` rs ON rs.id = mc.name_id
            AND rs.guid = mc.guid
        UNION ALL
        -- Regions
        SELECT
            rs.string AS name,
            (END - start) AS duration
        FROM
            `rocpd_region` rr
            INNER JOIN `rocpd_string` rs ON rs.id = rr.name_id
            AND rs.guid = rr.guid
    ) operations
    CROSS JOIN (
        SELECT
            SUM(END - start) AS total_time
        FROM
            (
                SELECT
                    END,
                    start
                FROM
                    `rocpd_kernel_dispatch`
                UNION ALL
                SELECT
                    END,
                    start
                FROM
                    `rocpd_memory_copy`
                UNION ALL
                SELECT
                    END,
                    start
                FROM
                    `rocpd_region`
            )
    ) TOTAL
GROUP BY
    name
ORDER BY
    total_duration DESC
/* top(name,total_calls,total_duration,average,percentage) */;
