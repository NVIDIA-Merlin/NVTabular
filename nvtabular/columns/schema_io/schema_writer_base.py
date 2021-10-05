class SchemaWriter():

    @classmethod
    def _read_scheam(cls, schema_path):
        raise NotImplementedError("Must have logic to read schema from file")

    @classmethod
    def write_schema(cls, schema, schema_path):
        raise NotImplementedError("Must have logic to write schema to file")

    @staticmethod
    def load_schema(cls, schema_path):
        raise NotImplementedError("logic to create schema object from file")

